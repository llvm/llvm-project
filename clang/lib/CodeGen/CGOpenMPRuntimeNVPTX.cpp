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

namespace {
enum OpenMPRTLFunctionNVPTX {
  /// \brief Call to void __kmpc_kernel_init(kmp_int32 thread_limit);
  OMPRTL_NVPTX__kmpc_kernel_init,
  /// \brief Call to void __kmpc_kernel_deinit();
  OMPRTL_NVPTX__kmpc_kernel_deinit,
  /// \brief Call to void __kmpc_spmd_kernel_init(kmp_int32 thread_limit,
  /// short RequiresOMPRuntime, short RequiresDataSharing);
  OMPRTL_NVPTX__kmpc_spmd_kernel_init,
  /// \brief Call to void __kmpc_spmd_kernel_deinit();
  OMPRTL_NVPTX__kmpc_spmd_kernel_deinit,
  /// \brief Call to void __kmpc_kernel_prepare_parallel(void
  /// *outlined_function);
  OMPRTL_NVPTX__kmpc_kernel_prepare_parallel,
  /// \brief Call to bool __kmpc_kernel_parallel(void **outlined_function);
  OMPRTL_NVPTX__kmpc_kernel_parallel,
  /// \brief Call to void __kmpc_kernel_end_parallel();
  OMPRTL_NVPTX__kmpc_kernel_end_parallel,
  /// Call to void __kmpc_serialized_parallel(ident_t *loc, kmp_int32
  /// global_tid);
  OMPRTL_NVPTX__kmpc_serialized_parallel,
  /// Call to void __kmpc_end_serialized_parallel(ident_t *loc, kmp_int32
  /// global_tid);
  OMPRTL_NVPTX__kmpc_end_serialized_parallel,
};

/// Pre(post)-action for different OpenMP constructs specialized for NVPTX.
class NVPTXActionTy final : public PrePostActionTy {
  llvm::Value *EnterCallee;
  ArrayRef<llvm::Value *> EnterArgs;
  llvm::Value *ExitCallee;
  ArrayRef<llvm::Value *> ExitArgs;
  bool Conditional;
  llvm::BasicBlock *ContBlock = nullptr;

public:
  NVPTXActionTy(llvm::Value *EnterCallee, ArrayRef<llvm::Value *> EnterArgs,
                llvm::Value *ExitCallee, ArrayRef<llvm::Value *> ExitArgs,
                bool Conditional = false)
      : EnterCallee(EnterCallee), EnterArgs(EnterArgs), ExitCallee(ExitCallee),
        ExitArgs(ExitArgs), Conditional(Conditional) {}
  void Enter(CodeGenFunction &CGF) override {
    llvm::Value *EnterRes = CGF.EmitRuntimeCall(EnterCallee, EnterArgs);
    if (Conditional) {
      llvm::Value *CallBool = CGF.Builder.CreateIsNotNull(EnterRes);
      auto *ThenBlock = CGF.createBasicBlock("omp_if.then");
      ContBlock = CGF.createBasicBlock("omp_if.end");
      // Generate the branch (If-stmt)
      CGF.Builder.CreateCondBr(CallBool, ThenBlock, ContBlock);
      CGF.EmitBlock(ThenBlock);
    }
  }
  void Done(CodeGenFunction &CGF) {
    // Emit the rest of blocks/branches
    CGF.EmitBranch(ContBlock);
    CGF.EmitBlock(ContBlock, true);
  }
  void Exit(CodeGenFunction &CGF) override {
    CGF.EmitRuntimeCall(ExitCallee, ExitArgs);
  }
};

// A class to track the execution mode when codegening directives within
// a target region. The appropriate mode (generic/spmd) is set on entry
// to the target region and used by containing directives such as 'parallel'
// to emit optimized code.
class ExecutionModeRAII {
private:
  CGOpenMPRuntimeNVPTX::ExecutionMode SavedMode;
  CGOpenMPRuntimeNVPTX::ExecutionMode &Mode;

public:
  ExecutionModeRAII(CGOpenMPRuntimeNVPTX::ExecutionMode &Mode,
                    CGOpenMPRuntimeNVPTX::ExecutionMode NewMode)
      : Mode(Mode) {
    SavedMode = Mode;
    Mode = NewMode;
  }
  ~ExecutionModeRAII() { Mode = SavedMode; }
};
} // anonymous namespace

/// Get the GPU warp size.
static llvm::Value *getNVPTXWarpSize(CodeGenFunction &CGF) {
  CGBuilderTy &Bld = CGF.Builder;
  return Bld.CreateCall(
      llvm::Intrinsic::getDeclaration(
          &CGF.CGM.getModule(), llvm::Intrinsic::nvvm_read_ptx_sreg_warpsize),
      llvm::None, "nvptx_warp_size");
}

/// Get the id of the current thread on the GPU.
static llvm::Value *getNVPTXThreadID(CodeGenFunction &CGF) {
  CGBuilderTy &Bld = CGF.Builder;
  return Bld.CreateCall(
      llvm::Intrinsic::getDeclaration(
          &CGF.CGM.getModule(), llvm::Intrinsic::nvvm_read_ptx_sreg_tid_x),
      llvm::None, "nvptx_tid");
}

/// Get the maximum number of threads in a block of the GPU.
static llvm::Value *getNVPTXNumThreads(CodeGenFunction &CGF) {
  CGBuilderTy &Bld = CGF.Builder;
  return Bld.CreateCall(
      llvm::Intrinsic::getDeclaration(
          &CGF.CGM.getModule(), llvm::Intrinsic::nvvm_read_ptx_sreg_ntid_x),
      llvm::None, "nvptx_num_threads");
}

/// Get barrier to synchronize all threads in a block.
static void getNVPTXCTABarrier(CodeGenFunction &CGF) {
  CGBuilderTy &Bld = CGF.Builder;
  Bld.CreateCall(llvm::Intrinsic::getDeclaration(
      &CGF.CGM.getModule(), llvm::Intrinsic::nvvm_barrier0));
}

/// Synchronize all GPU threads in a block.
static void syncCTAThreads(CodeGenFunction &CGF) { getNVPTXCTABarrier(CGF); }

/// Get the value of the thread_limit clause in the teams directive.
/// For the 'generic' execution mode, the runtime encodes thread_limit in
/// the launch parameters, always starting thread_limit+warpSize threads per
/// CTA. The threads in the last warp are reserved for master execution.
/// For the 'spmd' execution mode, all threads in a CTA are part of the team.
static llvm::Value *getThreadLimit(CodeGenFunction &CGF,
                                   bool IsInSpmdExecutionMode = false) {
  CGBuilderTy &Bld = CGF.Builder;
  return IsInSpmdExecutionMode
             ? getNVPTXNumThreads(CGF)
             : Bld.CreateSub(getNVPTXNumThreads(CGF), getNVPTXWarpSize(CGF),
                             "thread_limit");
}

/// Get the thread id of the OMP master thread.
/// The master thread id is the first thread (lane) of the last warp in the
/// GPU block.  Warp size is assumed to be some power of 2.
/// Thread id is 0 indexed.
/// E.g: If NumThreads is 33, master id is 32.
///      If NumThreads is 64, master id is 32.
///      If NumThreads is 1024, master id is 992.
static llvm::Value *getMasterThreadID(CodeGenFunction &CGF) {
  CGBuilderTy &Bld = CGF.Builder;
  llvm::Value *NumThreads = getNVPTXNumThreads(CGF);

  // We assume that the warp size is a power of 2.
  llvm::Value *Mask = Bld.CreateSub(getNVPTXWarpSize(CGF), Bld.getInt32(1));

  return Bld.CreateAnd(Bld.CreateSub(NumThreads, Bld.getInt32(1)),
                       Bld.CreateNot(Mask), "master_tid");
}

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
}

bool CGOpenMPRuntimeNVPTX::isInSpmdExecutionMode() const {
  return CurrentExecutionMode == CGOpenMPRuntimeNVPTX::ExecutionMode::Spmd;
}

static CGOpenMPRuntimeNVPTX::ExecutionMode
getExecutionModeForDirective(CodeGenModule &CGM,
                             const OMPExecutableDirective &D) {
  OpenMPDirectiveKind DirectiveKind = D.getDirectiveKind();
  switch (DirectiveKind) {
  case OMPD_target:
  case OMPD_target_teams:
    return CGOpenMPRuntimeNVPTX::ExecutionMode::Generic;
  case OMPD_target_parallel:
    return CGOpenMPRuntimeNVPTX::ExecutionMode::Spmd;
  default:
    llvm_unreachable("Unsupported directive on NVPTX device.");
  }
  llvm_unreachable("Unsupported directive on NVPTX device.");
}

void CGOpenMPRuntimeNVPTX::emitGenericKernel(const OMPExecutableDirective &D,
                                             StringRef ParentName,
                                             llvm::Function *&OutlinedFn,
                                             llvm::Constant *&OutlinedFnID,
                                             bool IsOffloadEntry,
                                             const RegionCodeGenTy &CodeGen) {
  ExecutionModeRAII ModeRAII(CurrentExecutionMode,
                             CGOpenMPRuntimeNVPTX::ExecutionMode::Generic);
  EntryFunctionState EST;
  WorkerFunctionState WST(CGM);
  Work.clear();

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
      RT.emitGenericEntryHeader(CGF, EST, WST);
    }
    void Exit(CodeGenFunction &CGF) override {
      RT.emitGenericEntryFooter(CGF, EST);
    }
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

// Setup NVPTX threads for master-worker OpenMP scheme.
void CGOpenMPRuntimeNVPTX::emitGenericEntryHeader(CodeGenFunction &CGF,
                                                  EntryFunctionState &EST,
                                                  WorkerFunctionState &WST) {
  CGBuilderTy &Bld = CGF.Builder;

  llvm::BasicBlock *WorkerBB = CGF.createBasicBlock(".worker");
  llvm::BasicBlock *MasterCheckBB = CGF.createBasicBlock(".mastercheck");
  llvm::BasicBlock *MasterBB = CGF.createBasicBlock(".master");
  EST.ExitBB = CGF.createBasicBlock(".exit");

  auto *IsWorker =
      Bld.CreateICmpULT(getNVPTXThreadID(CGF), getThreadLimit(CGF));
  Bld.CreateCondBr(IsWorker, WorkerBB, MasterCheckBB);

  CGF.EmitBlock(WorkerBB);
  CGF.EmitCallOrInvoke(WST.WorkerFn, llvm::None);
  CGF.EmitBranch(EST.ExitBB);

  CGF.EmitBlock(MasterCheckBB);
  auto *IsMaster =
      Bld.CreateICmpEQ(getNVPTXThreadID(CGF), getMasterThreadID(CGF));
  Bld.CreateCondBr(IsMaster, MasterBB, EST.ExitBB);

  CGF.EmitBlock(MasterBB);
  // First action in sequential region:
  // Initialize the state of the OpenMP runtime library on the GPU.
  llvm::Value *Args[] = {getThreadLimit(CGF)};
  CGF.EmitRuntimeCall(
      createNVPTXRuntimeFunction(OMPRTL_NVPTX__kmpc_kernel_init), Args);
}

void CGOpenMPRuntimeNVPTX::emitGenericEntryFooter(CodeGenFunction &CGF,
                                                  EntryFunctionState &EST) {
  if (!EST.ExitBB)
    EST.ExitBB = CGF.createBasicBlock(".exit");

  llvm::BasicBlock *TerminateBB = CGF.createBasicBlock(".termination.notifier");
  CGF.EmitBranch(TerminateBB);

  CGF.EmitBlock(TerminateBB);
  // Signal termination condition.
  CGF.EmitRuntimeCall(
      createNVPTXRuntimeFunction(OMPRTL_NVPTX__kmpc_kernel_deinit), None);
  // Barrier to terminate worker threads.
  syncCTAThreads(CGF);
  // Master thread jumps to exit point.
  CGF.EmitBranch(EST.ExitBB);

  CGF.EmitBlock(EST.ExitBB);
  EST.ExitBB = nullptr;
}

void CGOpenMPRuntimeNVPTX::emitSpmdKernel(const OMPExecutableDirective &D,
                                          StringRef ParentName,
                                          llvm::Function *&OutlinedFn,
                                          llvm::Constant *&OutlinedFnID,
                                          bool IsOffloadEntry,
                                          const RegionCodeGenTy &CodeGen) {
  ExecutionModeRAII ModeRAII(CurrentExecutionMode,
                             CGOpenMPRuntimeNVPTX::ExecutionMode::Spmd);
  EntryFunctionState EST;

  // Emit target region as a standalone region.
  class NVPTXPrePostActionTy : public PrePostActionTy {
    CGOpenMPRuntimeNVPTX &RT;
    CGOpenMPRuntimeNVPTX::EntryFunctionState &EST;
    const OMPExecutableDirective &D;

  public:
    NVPTXPrePostActionTy(CGOpenMPRuntimeNVPTX &RT,
                         CGOpenMPRuntimeNVPTX::EntryFunctionState &EST,
                         const OMPExecutableDirective &D)
        : RT(RT), EST(EST), D(D) {}
    void Enter(CodeGenFunction &CGF) override {
      RT.emitSpmdEntryHeader(CGF, EST, D);
    }
    void Exit(CodeGenFunction &CGF) override {
      RT.emitSpmdEntryFooter(CGF, EST);
    }
  } Action(*this, EST, D);
  CodeGen.setAction(Action);
  emitTargetOutlinedFunctionHelper(D, ParentName, OutlinedFn, OutlinedFnID,
                                   IsOffloadEntry, CodeGen);
  return;
}

void CGOpenMPRuntimeNVPTX::emitSpmdEntryHeader(
    CodeGenFunction &CGF, EntryFunctionState &EST,
    const OMPExecutableDirective &D) {
  auto &Bld = CGF.Builder;

  // Setup BBs in entry function.
  llvm::BasicBlock *ExecuteBB = CGF.createBasicBlock(".execute");
  EST.ExitBB = CGF.createBasicBlock(".exit");

  // Initialize the OMP state in the runtime; called by all active threads.
  // TODO: Set RequiresOMPRuntime and RequiresDataSharing parameters
  // based on code analysis of the target region.
  llvm::Value *Args[] = {getThreadLimit(CGF, /*IsInSpmdExecutionMode=*/true),
                         /*RequiresOMPRuntime=*/Bld.getInt16(1),
                         /*RequiresDataSharing=*/Bld.getInt16(1)};
  CGF.EmitRuntimeCall(
      createNVPTXRuntimeFunction(OMPRTL_NVPTX__kmpc_spmd_kernel_init), Args);
  CGF.EmitBranch(ExecuteBB);

  CGF.EmitBlock(ExecuteBB);
}

void CGOpenMPRuntimeNVPTX::emitSpmdEntryFooter(CodeGenFunction &CGF,
                                               EntryFunctionState &EST) {
  if (!EST.ExitBB)
    EST.ExitBB = CGF.createBasicBlock(".exit");

  llvm::BasicBlock *OMPDeInitBB = CGF.createBasicBlock(".omp.deinit");
  CGF.EmitBranch(OMPDeInitBB);

  CGF.EmitBlock(OMPDeInitBB);
  // DeInitialize the OMP state in the runtime; called by all active threads.
  CGF.EmitRuntimeCall(
      createNVPTXRuntimeFunction(OMPRTL_NVPTX__kmpc_spmd_kernel_deinit), None);
  CGF.EmitBranch(EST.ExitBB);

  CGF.EmitBlock(EST.ExitBB);
  EST.ExitBB = nullptr;
}

// Create a unique global variable to indicate the execution mode of this target
// region. The execution mode is either 'generic', or 'spmd' depending on the
// target directive. This variable is picked up by the offload library to setup
// the device appropriately before kernel launch. If the execution mode is
// 'generic', the runtime reserves one warp for the master, otherwise, all
// warps participate in parallel work.
static void setPropertyExecutionMode(CodeGenModule &CGM, StringRef Name,
                                     CGOpenMPRuntimeNVPTX::ExecutionMode Mode) {
  (void)new llvm::GlobalVariable(
      CGM.getModule(), CGM.Int8Ty, /*isConstant=*/true,
      llvm::GlobalValue::WeakAnyLinkage,
      llvm::ConstantInt::get(CGM.Int8Ty, Mode), Name + Twine("_exec_mode"));
}

void CGOpenMPRuntimeNVPTX::emitWorkerFunction(WorkerFunctionState &WST) {
  auto &Ctx = CGM.getContext();

  CodeGenFunction CGF(CGM, /*suppressNewContext=*/true);
  CGF.disableDebugInfo();
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

  Address WorkFn =
      CGF.CreateDefaultAlignTempAlloca(CGF.Int8PtrTy, /*Name=*/"work_fn");
  Address ExecStatus =
      CGF.CreateDefaultAlignTempAlloca(CGF.Int8Ty, /*Name=*/"exec_status");
  CGF.InitTempAlloca(ExecStatus, Bld.getInt8(/*C=*/0));
  CGF.InitTempAlloca(WorkFn, llvm::Constant::getNullValue(CGF.Int8PtrTy));

  llvm::Value *Args[] = {WorkFn.getPointer()};
  llvm::Value *Ret = CGF.EmitRuntimeCall(
      createNVPTXRuntimeFunction(OMPRTL_NVPTX__kmpc_kernel_parallel), Args);
  Bld.CreateStore(Bld.CreateZExt(Ret, CGF.Int8Ty), ExecStatus);

  // On termination condition (workid == 0), exit loop.
  llvm::Value *ShouldTerminate =
      Bld.CreateIsNull(Bld.CreateLoad(WorkFn), "should_terminate");
  Bld.CreateCondBr(ShouldTerminate, ExitBB, SelectWorkersBB);

  // Activate requested workers.
  CGF.EmitBlock(SelectWorkersBB);
  llvm::Value *IsActive =
      Bld.CreateIsNotNull(Bld.CreateLoad(ExecStatus), "is_active");
  Bld.CreateCondBr(IsActive, ExecuteBB, BarrierBB);

  // Signal start of parallel region.
  CGF.EmitBlock(ExecuteBB);

  // Process work items: outlined parallel functions.
  for (auto *W : Work) {
    // Try to match this outlined function.
    auto *ID = Bld.CreatePointerBitCastOrAddrSpaceCast(W, CGM.Int8PtrTy);

    llvm::Value *WorkFnMatch =
        Bld.CreateICmpEQ(Bld.CreateLoad(WorkFn), ID, "work_match");

    llvm::BasicBlock *ExecuteFNBB = CGF.createBasicBlock(".execute.fn");
    llvm::BasicBlock *CheckNextBB = CGF.createBasicBlock(".check.next");
    Bld.CreateCondBr(WorkFnMatch, ExecuteFNBB, CheckNextBB);

    // Execute this outlined function.
    CGF.EmitBlock(ExecuteFNBB);

    // Insert call to work function.
    // FIXME: Pass arguments to outlined function from master thread.
    auto *Fn = cast<llvm::Function>(W);
    Address ZeroAddr =
        CGF.CreateDefaultAlignTempAlloca(CGF.Int32Ty, /*Name=*/".zero.addr");
    CGF.InitTempAlloca(ZeroAddr, CGF.Builder.getInt32(/*C=*/0));
    llvm::Value *FnArgs[] = {ZeroAddr.getPointer(), ZeroAddr.getPointer()};
    CGF.EmitCallOrInvoke(Fn, FnArgs);

    // Go to end of parallel region.
    CGF.EmitBranch(TerminateBB);

    CGF.EmitBlock(CheckNextBB);
  }

  // Signal end of parallel region.
  CGF.EmitBlock(TerminateBB);
  CGF.EmitRuntimeCall(
      createNVPTXRuntimeFunction(OMPRTL_NVPTX__kmpc_kernel_end_parallel),
      llvm::None);
  CGF.EmitBranch(BarrierBB);

  // All active and inactive workers wait at a barrier after parallel region.
  CGF.EmitBlock(BarrierBB);
  // Barrier after parallel region.
  syncCTAThreads(CGF);
  CGF.EmitBranch(AwaitBB);

  // Exit target region.
  CGF.EmitBlock(ExitBB);
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
    // Build void __kmpc_kernel_init(kmp_int32 thread_limit);
    llvm::Type *TypeParams[] = {CGM.Int32Ty};
    llvm::FunctionType *FnTy =
        llvm::FunctionType::get(CGM.VoidTy, TypeParams, /*isVarArg*/ false);
    RTLFn = CGM.CreateRuntimeFunction(FnTy, "__kmpc_kernel_init");
    break;
  }
  case OMPRTL_NVPTX__kmpc_kernel_deinit: {
    // Build void __kmpc_kernel_deinit();
    llvm::FunctionType *FnTy =
        llvm::FunctionType::get(CGM.VoidTy, llvm::None, /*isVarArg*/ false);
    RTLFn = CGM.CreateRuntimeFunction(FnTy, "__kmpc_kernel_deinit");
    break;
  }
  case OMPRTL_NVPTX__kmpc_spmd_kernel_init: {
    // Build void __kmpc_spmd_kernel_init(kmp_int32 thread_limit,
    // short RequiresOMPRuntime, short RequiresDataSharing);
    llvm::Type *TypeParams[] = {CGM.Int32Ty, CGM.Int16Ty, CGM.Int16Ty};
    llvm::FunctionType *FnTy =
        llvm::FunctionType::get(CGM.VoidTy, TypeParams, /*isVarArg*/ false);
    RTLFn = CGM.CreateRuntimeFunction(FnTy, "__kmpc_spmd_kernel_init");
    break;
  }
  case OMPRTL_NVPTX__kmpc_spmd_kernel_deinit: {
    // Build void __kmpc_spmd_kernel_deinit();
    llvm::FunctionType *FnTy =
        llvm::FunctionType::get(CGM.VoidTy, llvm::None, /*isVarArg*/ false);
    RTLFn = CGM.CreateRuntimeFunction(FnTy, "__kmpc_spmd_kernel_deinit");
    break;
  }
  case OMPRTL_NVPTX__kmpc_kernel_prepare_parallel: {
    /// Build void __kmpc_kernel_prepare_parallel(
    /// void *outlined_function);
    llvm::Type *TypeParams[] = {CGM.Int8PtrTy};
    llvm::FunctionType *FnTy =
        llvm::FunctionType::get(CGM.VoidTy, TypeParams, /*isVarArg*/ false);
    RTLFn = CGM.CreateRuntimeFunction(FnTy, "__kmpc_kernel_prepare_parallel");
    break;
  }
  case OMPRTL_NVPTX__kmpc_kernel_parallel: {
    /// Build bool __kmpc_kernel_parallel(void **outlined_function);
    llvm::Type *TypeParams[] = {CGM.Int8PtrPtrTy};
    llvm::Type *RetTy = CGM.getTypes().ConvertType(CGM.getContext().BoolTy);
    llvm::FunctionType *FnTy =
        llvm::FunctionType::get(RetTy, TypeParams, /*isVarArg*/ false);
    RTLFn = CGM.CreateRuntimeFunction(FnTy, "__kmpc_kernel_parallel");
    break;
  }
  case OMPRTL_NVPTX__kmpc_kernel_end_parallel: {
    /// Build void __kmpc_kernel_end_parallel();
    llvm::FunctionType *FnTy =
        llvm::FunctionType::get(CGM.VoidTy, llvm::None, /*isVarArg*/ false);
    RTLFn = CGM.CreateRuntimeFunction(FnTy, "__kmpc_kernel_end_parallel");
    break;
  }
  case OMPRTL_NVPTX__kmpc_serialized_parallel: {
    // Build void __kmpc_serialized_parallel(ident_t *loc, kmp_int32
    // global_tid);
    llvm::Type *TypeParams[] = {getIdentTyPointerTy(), CGM.Int32Ty};
    llvm::FunctionType *FnTy =
        llvm::FunctionType::get(CGM.VoidTy, TypeParams, /*isVarArg*/ false);
    RTLFn = CGM.CreateRuntimeFunction(FnTy, "__kmpc_serialized_parallel");
    break;
  }
  case OMPRTL_NVPTX__kmpc_end_serialized_parallel: {
    // Build void __kmpc_end_serialized_parallel(ident_t *loc, kmp_int32
    // global_tid);
    llvm::Type *TypeParams[] = {getIdentTyPointerTy(), CGM.Int32Ty};
    llvm::FunctionType *FnTy =
        llvm::FunctionType::get(CGM.VoidTy, TypeParams, /*isVarArg*/ false);
    RTLFn = CGM.CreateRuntimeFunction(FnTy, "__kmpc_end_serialized_parallel");
    break;
  }
  }
  return RTLFn;
}

void CGOpenMPRuntimeNVPTX::createOffloadEntry(llvm::Constant *ID,
                                              llvm::Constant *Addr,
                                              uint64_t Size, int32_t) {
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

  CGOpenMPRuntimeNVPTX::ExecutionMode Mode =
      getExecutionModeForDirective(CGM, D);
  switch (Mode) {
  case CGOpenMPRuntimeNVPTX::ExecutionMode::Generic:
    emitGenericKernel(D, ParentName, OutlinedFn, OutlinedFnID, IsOffloadEntry,
                      CodeGen);
    break;
  case CGOpenMPRuntimeNVPTX::ExecutionMode::Spmd:
    emitSpmdKernel(D, ParentName, OutlinedFn, OutlinedFnID, IsOffloadEntry,
                   CodeGen);
    break;
  case CGOpenMPRuntimeNVPTX::ExecutionMode::Unknown:
    llvm_unreachable(
        "Unknown programming model for OpenMP directive on NVPTX target.");
  }

  setPropertyExecutionMode(CGM, OutlinedFn->getName(), Mode);
}

CGOpenMPRuntimeNVPTX::CGOpenMPRuntimeNVPTX(CodeGenModule &CGM)
    : CGOpenMPRuntime(CGM), CurrentExecutionMode(ExecutionMode::Unknown) {
  if (!CGM.getLangOpts().OpenMPIsDevice)
    llvm_unreachable("OpenMP NVPTX can only handle device code.");
}

void CGOpenMPRuntimeNVPTX::emitProcBindClause(CodeGenFunction &CGF,
                                              OpenMPProcBindClauseKind ProcBind,
                                              SourceLocation Loc) {
  // Do nothing in case of Spmd mode and L0 parallel.
  // TODO: If in Spmd mode and L1 parallel emit the clause.
  if (isInSpmdExecutionMode())
    return;

  CGOpenMPRuntime::emitProcBindClause(CGF, ProcBind, Loc);
}

void CGOpenMPRuntimeNVPTX::emitNumThreadsClause(CodeGenFunction &CGF,
                                                llvm::Value *NumThreads,
                                                SourceLocation Loc) {
  // Do nothing in case of Spmd mode and L0 parallel.
  // TODO: If in Spmd mode and L1 parallel emit the clause.
  if (isInSpmdExecutionMode())
    return;

  CGOpenMPRuntime::emitNumThreadsClause(CGF, NumThreads, Loc);
}

void CGOpenMPRuntimeNVPTX::emitNumTeamsClause(CodeGenFunction &CGF,
                                              const Expr *NumTeams,
                                              const Expr *ThreadLimit,
                                              SourceLocation Loc) {}

llvm::Value *CGOpenMPRuntimeNVPTX::emitParallelOutlinedFunction(
    const OMPExecutableDirective &D, const VarDecl *ThreadIDVar,
    OpenMPDirectiveKind InnermostKind, const RegionCodeGenTy &CodeGen) {
  return CGOpenMPRuntime::emitParallelOutlinedFunction(D, ThreadIDVar,
                                                       InnermostKind, CodeGen);
}

llvm::Value *CGOpenMPRuntimeNVPTX::emitTeamsOutlinedFunction(
    const OMPExecutableDirective &D, const VarDecl *ThreadIDVar,
    OpenMPDirectiveKind InnermostKind, const RegionCodeGenTy &CodeGen) {

  llvm::Value *OutlinedFunVal = CGOpenMPRuntime::emitTeamsOutlinedFunction(
      D, ThreadIDVar, InnermostKind, CodeGen);
  llvm::Function *OutlinedFun = cast<llvm::Function>(OutlinedFunVal);
  OutlinedFun->removeFnAttr(llvm::Attribute::NoInline);
  OutlinedFun->addFnAttr(llvm::Attribute::AlwaysInline);

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

void CGOpenMPRuntimeNVPTX::emitParallelCall(
    CodeGenFunction &CGF, SourceLocation Loc, llvm::Value *OutlinedFn,
    ArrayRef<llvm::Value *> CapturedVars, const Expr *IfCond) {
  if (!CGF.HaveInsertPoint())
    return;

  if (isInSpmdExecutionMode())
    emitSpmdParallelCall(CGF, Loc, OutlinedFn, CapturedVars, IfCond);
  else
    emitGenericParallelCall(CGF, Loc, OutlinedFn, CapturedVars, IfCond);
}

void CGOpenMPRuntimeNVPTX::emitGenericParallelCall(
    CodeGenFunction &CGF, SourceLocation Loc, llvm::Value *OutlinedFn,
    ArrayRef<llvm::Value *> CapturedVars, const Expr *IfCond) {
  llvm::Function *Fn = cast<llvm::Function>(OutlinedFn);

  auto &&L0ParallelGen = [this, Fn](CodeGenFunction &CGF, PrePostActionTy &) {
    CGBuilderTy &Bld = CGF.Builder;

    // Prepare for parallel region. Indicate the outlined function.
    llvm::Value *Args[] = {Bld.CreateBitOrPointerCast(Fn, CGM.Int8PtrTy)};
    CGF.EmitRuntimeCall(
        createNVPTXRuntimeFunction(OMPRTL_NVPTX__kmpc_kernel_prepare_parallel),
        Args);

    // Activate workers. This barrier is used by the master to signal
    // work for the workers.
    syncCTAThreads(CGF);

    // OpenMP [2.5, Parallel Construct, p.49]
    // There is an implied barrier at the end of a parallel region. After the
    // end of a parallel region, only the master thread of the team resumes
    // execution of the enclosing task region.
    //
    // The master waits at this barrier until all workers are done.
    syncCTAThreads(CGF);

    // Remember for post-processing in worker loop.
    Work.push_back(Fn);
  };

  auto *RTLoc = emitUpdateLocation(CGF, Loc);
  auto *ThreadID = getThreadID(CGF, Loc);
  llvm::Value *Args[] = {RTLoc, ThreadID};

  auto &&SeqGen = [this, Fn, &CapturedVars, &Args](CodeGenFunction &CGF,
                                                   PrePostActionTy &) {
    auto &&CodeGen = [this, Fn, &CapturedVars](CodeGenFunction &CGF,
                                               PrePostActionTy &Action) {
      Action.Enter(CGF);

      llvm::SmallVector<llvm::Value *, 16> OutlinedFnArgs;
      OutlinedFnArgs.push_back(
          llvm::ConstantPointerNull::get(CGM.Int32Ty->getPointerTo()));
      OutlinedFnArgs.push_back(
          llvm::ConstantPointerNull::get(CGM.Int32Ty->getPointerTo()));
      OutlinedFnArgs.append(CapturedVars.begin(), CapturedVars.end());
      CGF.EmitCallOrInvoke(Fn, OutlinedFnArgs);
    };

    RegionCodeGenTy RCG(CodeGen);
    NVPTXActionTy Action(
        createNVPTXRuntimeFunction(OMPRTL_NVPTX__kmpc_serialized_parallel),
        Args,
        createNVPTXRuntimeFunction(OMPRTL_NVPTX__kmpc_end_serialized_parallel),
        Args);
    RCG.setAction(Action);
    RCG(CGF);
  };

  if (IfCond)
    emitOMPIfClause(CGF, IfCond, L0ParallelGen, SeqGen);
  else {
    CodeGenFunction::RunCleanupsScope Scope(CGF);
    RegionCodeGenTy ThenRCG(L0ParallelGen);
    ThenRCG(CGF);
  }
}

void CGOpenMPRuntimeNVPTX::emitSpmdParallelCall(
    CodeGenFunction &CGF, SourceLocation Loc, llvm::Value *OutlinedFn,
    ArrayRef<llvm::Value *> CapturedVars, const Expr *IfCond) {
  // Just call the outlined function to execute the parallel region.
  // OutlinedFn(&GTid, &zero, CapturedStruct);
  //
  // TODO: Do something with IfCond when support for the 'if' clause
  // is added on Spmd target directives.
  llvm::SmallVector<llvm::Value *, 16> OutlinedFnArgs;
  OutlinedFnArgs.push_back(
      llvm::ConstantPointerNull::get(CGM.Int32Ty->getPointerTo()));
  OutlinedFnArgs.push_back(
      llvm::ConstantPointerNull::get(CGM.Int32Ty->getPointerTo()));
  OutlinedFnArgs.append(CapturedVars.begin(), CapturedVars.end());
  CGF.EmitCallOrInvoke(OutlinedFn, OutlinedFnArgs);
}
