//===---- CGOpenMPRuntimeNVPTX.cpp - Interface to OpenMP NVPTX Runtimes ---===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This provides a class for OpenMP runtime code generation specialized to NVPTX
// targets.
//
//===----------------------------------------------------------------------===//

#include "CGOpenMPRuntimeNVPTX.h"
#include "clang/AST/DeclOpenMP.h"
#include "CodeGenFunction.h"
#include "clang/AST/StmtOpenMP.h"

using namespace clang;
using namespace CodeGen;

/// \brief Get the GPU warp size.
llvm::Value *CGOpenMPRuntimeNVPTX::getNVPTXWarpSize(CodeGenFunction &CGF) {
  CGBuilderTy &Bld = CGF.Builder;
  return Bld.CreateCall(
      llvm::Intrinsic::getDeclaration(
          &CGM.getModule(), llvm::Intrinsic::nvvm_read_ptx_sreg_warpsize),
      llvm::None, "nvptx_warp_size");
}

/// \brief Get the id of the current thread on the GPU.
llvm::Value *CGOpenMPRuntimeNVPTX::getNVPTXThreadID(CodeGenFunction &CGF) {
  CGBuilderTy &Bld = CGF.Builder;
  return Bld.CreateCall(
      llvm::Intrinsic::getDeclaration(
          &CGM.getModule(), llvm::Intrinsic::nvvm_read_ptx_sreg_tid_x),
      llvm::None, "nvptx_tid");
}

// \brief Get the maximum number of threads in a block of the GPU.
llvm::Value *CGOpenMPRuntimeNVPTX::getNVPTXNumThreads(CodeGenFunction &CGF) {
  CGBuilderTy &Bld = CGF.Builder;
  return Bld.CreateCall(
      llvm::Intrinsic::getDeclaration(
          &CGM.getModule(), llvm::Intrinsic::nvvm_read_ptx_sreg_ntid_x),
      llvm::None, "nvptx_num_threads");
}

/// \brief Get barrier to synchronize all threads in a block.
void CGOpenMPRuntimeNVPTX::getNVPTXCTABarrier(CodeGenFunction &CGF) {
  CGBuilderTy &Bld = CGF.Builder;
  Bld.CreateCall(llvm::Intrinsic::getDeclaration(
      &CGM.getModule(), llvm::Intrinsic::nvvm_barrier0));
}

// \brief Synchronize all GPU threads in a block.
void CGOpenMPRuntimeNVPTX::syncCTAThreads(CodeGenFunction &CGF) {
  getNVPTXCTABarrier(CGF);
}

/// \brief Get the thread id of the OMP master thread.
/// The master thread id is the first thread (lane) of the last warp in the
/// GPU block.  Warp size is assumed to be some power of 2.
/// Thread id is 0 indexed.
/// E.g: If NumThreads is 33, master id is 32.
///      If NumThreads is 64, master id is 32.
///      If NumThreads is 1024, master id is 992.
llvm::Value *CGOpenMPRuntimeNVPTX::getMasterThreadID(CodeGenFunction &CGF) {
  CGBuilderTy &Bld = CGF.Builder;
  llvm::Value *NumThreads = getNVPTXNumThreads(CGF);

  // We assume that the warp size is a power of 2.
  llvm::Value *Mask = Bld.CreateSub(getNVPTXWarpSize(CGF), Bld.getInt32(1));

  return Bld.CreateAnd(Bld.CreateSub(NumThreads, Bld.getInt32(1)),
                       Bld.CreateNot(Mask), "master_tid");
}

namespace {
enum OpenMPRTLFunctionNVPTX {
  /// \brief Call to void __kmpc_kernel_init(kmp_int32 omp_handle,
  /// kmp_int32 thread_limit);
  OMPRTL_NVPTX__kmpc_kernel_init,
};

// NVPTX Address space
enum ADDRESS_SPACE {
  ADDRESS_SPACE_SHARED = 3,
};
} // namespace

CGOpenMPRuntimeNVPTX::WorkerFunctionState::WorkerFunctionState(
    CodeGenModule &CGM)
    : WorkerFn(nullptr), CGFI(nullptr) {
  createWorkerFunction(CGM);
}

void CGOpenMPRuntimeNVPTX::WorkerFunctionState::createWorkerFunction(
    CodeGenModule &CGM) {
  // Create an worker function with no arguments.
  CGFI = &CGM.getTypes().arrangeNullaryFunction();

  WorkerFn = llvm::Function::Create(
      CGM.getTypes().GetFunctionType(*CGFI), llvm::GlobalValue::InternalLinkage,
      /* placeholder */ "_worker", &CGM.getModule());
  CGM.SetInternalFunctionAttributes(/*D=*/nullptr, WorkerFn, *CGFI);
  WorkerFn->setLinkage(llvm::GlobalValue::InternalLinkage);
  WorkerFn->addFnAttr(llvm::Attribute::NoInline);
}

void CGOpenMPRuntimeNVPTX::initializeEnvironment() {
  //
  // Initialize master-worker control state in shared memory.
  //

  auto DL = CGM.getDataLayout();
  ActiveWorkers = new llvm::GlobalVariable(
      CGM.getModule(), CGM.Int32Ty, /*isConstant=*/false,
      llvm::GlobalValue::CommonLinkage,
      llvm::Constant::getNullValue(CGM.Int32Ty), "__omp_num_threads", 0,
      llvm::GlobalVariable::NotThreadLocal, ADDRESS_SPACE_SHARED);
  ActiveWorkers->setAlignment(DL.getPrefTypeAlignment(CGM.Int32Ty));

  WorkID = new llvm::GlobalVariable(
      CGM.getModule(), CGM.Int64Ty, /*isConstant=*/false,
      llvm::GlobalValue::CommonLinkage,
      llvm::Constant::getNullValue(CGM.Int64Ty), "__tgt_work_id", 0,
      llvm::GlobalVariable::NotThreadLocal, ADDRESS_SPACE_SHARED);
  WorkID->setAlignment(DL.getPrefTypeAlignment(CGM.Int64Ty));
}

void CGOpenMPRuntimeNVPTX::emitWorkerFunction(WorkerFunctionState &WST) {
  auto &Ctx = CGM.getContext();

  CodeGenFunction CGF(CGM, /*suppressNewContext=*/true);
  CGF.StartFunction(GlobalDecl(), Ctx.VoidTy, WST.WorkerFn, *WST.CGFI, {});
  emitWorkerLoop(CGF, WST);
  CGF.FinishFunction();
}

void CGOpenMPRuntimeNVPTX::emitWorkerLoop(CodeGenFunction &CGF,
                                          WorkerFunctionState &WST) {
  //
  // The workers enter this loop and wait for parallel work from the master.
  // When the master encounters a parallel region it sets up the work + variable
  // arguments, and wakes up the workers.  The workers first check to see if
  // they are required for the parallel region, i.e., within the # of requested
  // parallel threads.  The activated workers load the variable arguments and
  // execute the parallel work.
  //

  CGBuilderTy &Bld = CGF.Builder;

  llvm::BasicBlock *AwaitBB = CGF.createBasicBlock(".await.work");
  llvm::BasicBlock *SelectWorkersBB = CGF.createBasicBlock(".select.workers");
  llvm::BasicBlock *ExecuteBB = CGF.createBasicBlock(".execute.parallel");
  llvm::BasicBlock *TerminateBB = CGF.createBasicBlock(".terminate.parallel");
  llvm::BasicBlock *BarrierBB = CGF.createBasicBlock(".barrier.parallel");
  llvm::BasicBlock *ExitBB = CGF.createBasicBlock(".exit");

  CGF.EmitBranch(AwaitBB);

  // Workers wait for work from master.
  CGF.EmitBlock(AwaitBB);
  // Wait for parallel work
  syncCTAThreads(CGF);
  // On termination condition (workid == 0), exit loop.
  llvm::Value *ShouldTerminate = Bld.CreateICmpEQ(
      Bld.CreateAlignedLoad(WorkID, WorkID->getAlignment()),
      llvm::Constant::getNullValue(WorkID->getType()->getElementType()),
      "should_terminate");
  Bld.CreateCondBr(ShouldTerminate, ExitBB, SelectWorkersBB);

  // Activate requested workers.
  CGF.EmitBlock(SelectWorkersBB);
  llvm::Value *ThreadID = getNVPTXThreadID(CGF);
  llvm::Value *ActiveThread = Bld.CreateICmpSLT(
      ThreadID,
      Bld.CreateAlignedLoad(ActiveWorkers, ActiveWorkers->getAlignment()),
      "active_thread");
  Bld.CreateCondBr(ActiveThread, ExecuteBB, BarrierBB);

  // Signal start of parallel region.
  CGF.EmitBlock(ExecuteBB);
  // TODO: Add parallel work.

  // Signal end of parallel region.
  CGF.EmitBlock(TerminateBB);
  CGF.EmitBranch(BarrierBB);

  // All active and inactive workers wait at a barrier after parallel region.
  CGF.EmitBlock(BarrierBB);
  // Barrier after parallel region.
  syncCTAThreads(CGF);
  CGF.EmitBranch(AwaitBB);

  // Exit target region.
  CGF.EmitBlock(ExitBB);
}

// Setup NVPTX threads for master-worker OpenMP scheme.
void CGOpenMPRuntimeNVPTX::emitEntryHeader(CodeGenFunction &CGF,
                                           EntryFunctionState &EST,
                                           WorkerFunctionState &WST) {
  CGBuilderTy &Bld = CGF.Builder;

  // Get the master thread id.
  llvm::Value *MasterID = getMasterThreadID(CGF);
  // Current thread's identifier.
  llvm::Value *ThreadID = getNVPTXThreadID(CGF);

  // Setup BBs in entry function.
  llvm::BasicBlock *WorkerCheckBB = CGF.createBasicBlock(".check.for.worker");
  llvm::BasicBlock *WorkerBB = CGF.createBasicBlock(".worker");
  llvm::BasicBlock *MasterBB = CGF.createBasicBlock(".master");
  EST.ExitBB = CGF.createBasicBlock(".exit");

  // The head (master thread) marches on while its body of companion threads in
  // the warp go to sleep.
  llvm::Value *ShouldDie =
      Bld.CreateICmpUGT(ThreadID, MasterID, "excess_in_master_warp");
  Bld.CreateCondBr(ShouldDie, EST.ExitBB, WorkerCheckBB);

  // Select worker threads...
  CGF.EmitBlock(WorkerCheckBB);
  llvm::Value *IsWorker = Bld.CreateICmpULT(ThreadID, MasterID, "is_worker");
  Bld.CreateCondBr(IsWorker, WorkerBB, MasterBB);

  // ... and send to worker loop, awaiting parallel invocation.
  CGF.EmitBlock(WorkerBB);
  CGF.EmitCallOrInvoke(WST.WorkerFn, llvm::None);
  CGF.EmitBranch(EST.ExitBB);

  // Only master thread executes subsequent serial code.
  CGF.EmitBlock(MasterBB);

  // First action in sequential region:
  // Initialize the state of the OpenMP runtime library on the GPU.
  llvm::Value *Args[] = {Bld.getInt32(/*OmpHandle=*/0), getNVPTXThreadID(CGF)};
  CGF.EmitRuntimeCall(createNVPTXRuntimeFunction(OMPRTL_NVPTX__kmpc_kernel_init),
                      Args);
}

void CGOpenMPRuntimeNVPTX::emitEntryFooter(CodeGenFunction &CGF,
                                           EntryFunctionState &EST) {
  CGBuilderTy &Bld = CGF.Builder;
  llvm::BasicBlock *TerminateBB = CGF.createBasicBlock(".termination.notifier");
  CGF.EmitBranch(TerminateBB);

  CGF.EmitBlock(TerminateBB);
  // Signal termination condition.
  Bld.CreateAlignedStore(
      llvm::Constant::getNullValue(WorkID->getType()->getElementType()), WorkID,
      WorkID->getAlignment());
  // Barrier to terminate worker threads.
  syncCTAThreads(CGF);
  // Master thread jumps to exit point.
  CGF.EmitBranch(EST.ExitBB);

  CGF.EmitBlock(EST.ExitBB);
}

/// \brief Returns specified OpenMP runtime function for the current OpenMP
/// implementation.  Specialized for the NVPTX device.
/// \param Function OpenMP runtime function.
/// \return Specified function.
llvm::Constant *
CGOpenMPRuntimeNVPTX::createNVPTXRuntimeFunction(unsigned Function) {
  llvm::Constant *RTLFn = nullptr;
  switch (static_cast<OpenMPRTLFunctionNVPTX>(Function)) {
  case OMPRTL_NVPTX__kmpc_kernel_init: {
    // Build void __kmpc_kernel_init(kmp_int32 omp_handle,
    // kmp_int32 thread_limit);
    llvm::Type *TypeParams[] = {CGM.Int32Ty, CGM.Int32Ty};
    llvm::FunctionType *FnTy =
        llvm::FunctionType::get(CGM.VoidTy, TypeParams, /*isVarArg*/ false);
    RTLFn = CGM.CreateRuntimeFunction(FnTy, "__kmpc_kernel_init");
    break;
  }
  }
  return RTLFn;
}

void CGOpenMPRuntimeNVPTX::createOffloadEntry(llvm::Constant *ID,
                                              llvm::Constant *Addr,
                                              uint64_t Size) {
  auto *F = dyn_cast<llvm::Function>(Addr);
  // TODO: Add support for global variables on the device after declare target
  // support.
  if (!F)
    return;
  llvm::Module *M = F->getParent();
  llvm::LLVMContext &Ctx = M->getContext();

  // Get "nvvm.annotations" metadata node
  llvm::NamedMDNode *MD = M->getOrInsertNamedMetadata("nvvm.annotations");

  llvm::Metadata *MDVals[] = {
      llvm::ConstantAsMetadata::get(F), llvm::MDString::get(Ctx, "kernel"),
      llvm::ConstantAsMetadata::get(
          llvm::ConstantInt::get(llvm::Type::getInt32Ty(Ctx), 1))};
  // Append metadata to nvvm.annotations
  MD->addOperand(llvm::MDNode::get(Ctx, MDVals));
}

void CGOpenMPRuntimeNVPTX::emitTargetOutlinedFunction(
    const OMPExecutableDirective &D, StringRef ParentName,
    llvm::Function *&OutlinedFn, llvm::Constant *&OutlinedFnID,
    bool IsOffloadEntry, const RegionCodeGenTy &CodeGen) {
  if (!IsOffloadEntry) // Nothing to do.
    return;

  assert(!ParentName.empty() && "Invalid target region parent name!");

  EntryFunctionState EST;
  WorkerFunctionState WST(CGM);

  // Emit target region as a standalone region.
  class NVPTXPrePostActionTy : public PrePostActionTy {
    CGOpenMPRuntimeNVPTX &RT;
    CGOpenMPRuntimeNVPTX::EntryFunctionState &EST;
    CGOpenMPRuntimeNVPTX::WorkerFunctionState &WST;

  public:
    NVPTXPrePostActionTy(CGOpenMPRuntimeNVPTX &RT,
                         CGOpenMPRuntimeNVPTX::EntryFunctionState &EST,
                         CGOpenMPRuntimeNVPTX::WorkerFunctionState &WST)
        : RT(RT), EST(EST), WST(WST) {}
    void Enter(CodeGenFunction &CGF) override {
      RT.emitEntryHeader(CGF, EST, WST);
    }
    void Exit(CodeGenFunction &CGF) override { RT.emitEntryFooter(CGF, EST); }
  } Action(*this, EST, WST);
  CodeGen.setAction(Action);
  emitTargetOutlinedFunctionHelper(D, ParentName, OutlinedFn, OutlinedFnID,
                                   IsOffloadEntry, CodeGen);

  // Create the worker function
  emitWorkerFunction(WST);

  // Now change the name of the worker function to correspond to this target
  // region's entry function.
  WST.WorkerFn->setName(OutlinedFn->getName() + "_worker");
}

CGOpenMPRuntimeNVPTX::CGOpenMPRuntimeNVPTX(CodeGenModule &CGM)
    : CGOpenMPRuntime(CGM), ActiveWorkers(nullptr), WorkID(nullptr) {
  if (!CGM.getLangOpts().OpenMPIsDevice)
    llvm_unreachable("OpenMP NVPTX can only handle device code.");

  // Called once per module during initialization.
  initializeEnvironment();
}

void CGOpenMPRuntimeNVPTX::emitNumTeamsClause(CodeGenFunction &CGF,
                                              const Expr *NumTeams,
                                              const Expr *ThreadLimit,
                                              SourceLocation Loc) {}

llvm::Value *CGOpenMPRuntimeNVPTX::emitParallelOrTeamsOutlinedFunction(
    const OMPExecutableDirective &D, const VarDecl *ThreadIDVar,
    OpenMPDirectiveKind InnermostKind, const RegionCodeGenTy &CodeGen) {

  llvm::Function *OutlinedFun = nullptr;
  if (isa<OMPTeamsDirective>(D)) {
    llvm::Value *OutlinedFunVal =
        CGOpenMPRuntime::emitParallelOrTeamsOutlinedFunction(
            D, ThreadIDVar, InnermostKind, CodeGen);
    OutlinedFun = cast<llvm::Function>(OutlinedFunVal);
    OutlinedFun->addFnAttr(llvm::Attribute::AlwaysInline);
  } else
    llvm_unreachable("parallel directive is not yet supported for nvptx "
                     "backend.");

  return OutlinedFun;
}

void CGOpenMPRuntimeNVPTX::emitTeamsCall(CodeGenFunction &CGF,
                                         const OMPExecutableDirective &D,
                                         SourceLocation Loc,
                                         llvm::Value *OutlinedFn,
                                         ArrayRef<llvm::Value *> CapturedVars) {
  if (!CGF.HaveInsertPoint())
    return;

  Address ZeroAddr =
      CGF.CreateTempAlloca(CGF.Int32Ty, CharUnits::fromQuantity(4),
                           /*Name*/ ".zero.addr");
  CGF.InitTempAlloca(ZeroAddr, CGF.Builder.getInt32(/*C*/ 0));
  llvm::SmallVector<llvm::Value *, 16> OutlinedFnArgs;
  OutlinedFnArgs.push_back(ZeroAddr.getPointer());
  OutlinedFnArgs.push_back(ZeroAddr.getPointer());
  OutlinedFnArgs.append(CapturedVars.begin(), CapturedVars.end());
  CGF.EmitCallOrInvoke(OutlinedFn, OutlinedFnArgs);
}
