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
  /// \brief Call to void __kmpc_kernel_init(kmp_int32 thread_limit,
  /// int16_t RequiresOMPRuntime);
  OMPRTL_NVPTX__kmpc_kernel_init,
  /// \brief Call to void __kmpc_kernel_deinit(int16_t IsOMPRuntimeInitialized);
  OMPRTL_NVPTX__kmpc_kernel_deinit,
  /// \brief Call to void __kmpc_spmd_kernel_init(kmp_int32 thread_limit,
  /// int16_t RequiresOMPRuntime, int16_t RequiresDataSharing);
  OMPRTL_NVPTX__kmpc_spmd_kernel_init,
  /// \brief Call to void __kmpc_spmd_kernel_deinit();
  OMPRTL_NVPTX__kmpc_spmd_kernel_deinit,
  /// \brief Call to void __kmpc_kernel_prepare_parallel(void
  /// *outlined_function, void ***args, kmp_int32 nArgs, int16_t
  /// IsOMPRuntimeInitialized);
  OMPRTL_NVPTX__kmpc_kernel_prepare_parallel,
  /// \brief Call to bool __kmpc_kernel_parallel(void **outlined_function, void
  /// ***args, int16_t IsOMPRuntimeInitialized);
  OMPRTL_NVPTX__kmpc_kernel_parallel,
  /// \brief Call to void __kmpc_kernel_end_parallel();
  OMPRTL_NVPTX__kmpc_kernel_end_parallel,
  /// Call to void __kmpc_serialized_parallel(ident_t *loc, kmp_int32
  /// global_tid);
  OMPRTL_NVPTX__kmpc_serialized_parallel,
  /// Call to void __kmpc_end_serialized_parallel(ident_t *loc, kmp_int32
  /// global_tid);
  OMPRTL_NVPTX__kmpc_end_serialized_parallel,
  /// \brief Call to int32_t __kmpc_shuffle_int32(int32_t element,
  /// int16_t lane_offset, int16_t warp_size);
  OMPRTL_NVPTX__kmpc_shuffle_int32,
  /// \brief Call to int64_t __kmpc_shuffle_int64(int64_t element,
  /// int16_t lane_offset, int16_t warp_size);
  OMPRTL_NVPTX__kmpc_shuffle_int64,
  /// \brief Call to __kmpc_nvptx_parallel_reduce_nowait(kmp_int32
  /// global_tid, kmp_int32 num_vars, size_t reduce_size, void* reduce_data,
  /// void (*kmp_ShuffleReductFctPtr)(void *rhsData, int16_t lane_id, int16_t
  /// lane_offset, int16_t shortCircuit),
  /// void (*kmp_InterWarpCopyFctPtr)(void* src, int32_t warp_num));
  OMPRTL_NVPTX__kmpc_parallel_reduce_nowait,
  /// \brief Call to __kmpc_nvptx_teams_reduce_nowait(int32_t global_tid,
  /// int32_t num_vars, size_t reduce_size, void *reduce_data,
  /// void (*kmp_ShuffleReductFctPtr)(void *rhs, int16_t lane_id, int16_t
  /// lane_offset, int16_t shortCircuit),
  /// void (*kmp_InterWarpCopyFctPtr)(void* src, int32_t warp_num),
  /// void (*kmp_CopyToScratchpadFctPtr)(void *reduce_data, void * scratchpad,
  /// int32_t index, int32_t width),
  /// void (*kmp_LoadReduceFctPtr)(void *reduce_data, void * scratchpad, int32_t
  /// index, int32_t width, int32_t reduce))
  OMPRTL_NVPTX__kmpc_teams_reduce_nowait,
  /// \brief Call to __kmpc_nvptx_end_reduce_nowait(int32_t global_tid);
  OMPRTL_NVPTX__kmpc_end_reduce_nowait
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

/// GPU Configuration:  This information can be derived from cuda registers,
/// however, providing compile time constants helps generate more efficient
/// code.  For all practical purposes this is fine because the configuration
/// is the same for all known NVPTX architectures.
enum MachineConfiguration : unsigned {
  WarpSize = 32,
  /// Number of bits required to represent a lane identifier, which is
  /// computed as log_2(WarpSize).
  LaneIDBits = 5,
  LaneIDMask = WarpSize - 1,

  /// Global memory alignment for performance.
  GlobalMemoryAlignment = 256,
};

enum NamedBarrier : unsigned {
  /// Synchronize on this barrier #ID using a named barrier primitive.
  /// Only the subset of active threads in a parallel region arrive at the
  /// barrier.
  NB_Parallel = 1,
};
} // anonymous namespace

/// Get the GPU warp size.
static llvm::Value *getNVPTXWarpSize(CodeGenFunction &CGF) {
  return CGF.EmitRuntimeCall(
      llvm::Intrinsic::getDeclaration(
          &CGF.CGM.getModule(), llvm::Intrinsic::nvvm_read_ptx_sreg_warpsize),
      "nvptx_warp_size");
}

/// Get the id of the current thread on the GPU.
static llvm::Value *getNVPTXThreadID(CodeGenFunction &CGF) {
  return CGF.EmitRuntimeCall(
      llvm::Intrinsic::getDeclaration(
          &CGF.CGM.getModule(), llvm::Intrinsic::nvvm_read_ptx_sreg_tid_x),
      "nvptx_tid");
}

/// Get the id of the warp in the block.
/// We assume that the warp size is 32, which is always the case
/// on the NVPTX device, to generate more efficient code.
static llvm::Value *getNVPTXWarpID(CodeGenFunction &CGF) {
  CGBuilderTy &Bld = CGF.Builder;
  return Bld.CreateAShr(getNVPTXThreadID(CGF), LaneIDBits, "nvptx_warp_id");
}

/// Get the id of the current lane in the Warp.
/// We assume that the warp size is 32, which is always the case
/// on the NVPTX device, to generate more efficient code.
static llvm::Value *getNVPTXLaneID(CodeGenFunction &CGF) {
  CGBuilderTy &Bld = CGF.Builder;
  return Bld.CreateAnd(getNVPTXThreadID(CGF), Bld.getInt32(LaneIDMask),
                       "nvptx_lane_id");
}

/// Get the maximum number of threads in a block of the GPU.
static llvm::Value *getNVPTXNumThreads(CodeGenFunction &CGF) {
  return CGF.EmitRuntimeCall(
      llvm::Intrinsic::getDeclaration(
          &CGF.CGM.getModule(), llvm::Intrinsic::nvvm_read_ptx_sreg_ntid_x),
      "nvptx_num_threads");
}

/// Get barrier to synchronize all threads in a block.
static void getNVPTXCTABarrier(CodeGenFunction &CGF) {
  CGF.EmitRuntimeCall(llvm::Intrinsic::getDeclaration(
      &CGF.CGM.getModule(), llvm::Intrinsic::nvvm_barrier0));
}

/// Get barrier #ID to synchronize selected (multiple of warp size) threads in
/// a CTA.
static void getNVPTXBarrier(CodeGenFunction &CGF, int ID,
                            llvm::Value *NumThreads) {
  CGBuilderTy &Bld = CGF.Builder;
  llvm::Value *Args[] = {Bld.getInt32(ID), NumThreads};
  CGF.EmitRuntimeCall(llvm::Intrinsic::getDeclaration(
                          &CGF.CGM.getModule(), llvm::Intrinsic::nvvm_barrier),
                      Args);
}

/// Synchronize all GPU threads in a block.
static void syncCTAThreads(CodeGenFunction &CGF) { getNVPTXCTABarrier(CGF); }

/// Synchronize worker threads in a parallel region.
static void syncParallelThreads(CodeGenFunction &CGF, llvm::Value *NumThreads) {
  return getNVPTXBarrier(CGF, NB_Parallel, NumThreads);
}

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
    CodeGenModule &CGM, SourceLocation Loc)
    : WorkerFn(nullptr), CGFI(nullptr), Loc(Loc) {
  createWorkerFunction(CGM);
}

void CGOpenMPRuntimeNVPTX::WorkerFunctionState::createWorkerFunction(
    CodeGenModule &CGM) {
  // Create an worker function with no arguments.
  CGFI = &CGM.getTypes().arrangeNullaryFunction();

  WorkerFn = llvm::Function::Create(
      CGM.getTypes().GetFunctionType(*CGFI), llvm::GlobalValue::InternalLinkage,
      /*placeholder=*/"_worker", &CGM.getModule());
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
  case OMPD_target_parallel_for:
  case OMPD_target_parallel_for_simd:
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
  WorkerFunctionState WST(CGM, D.getLocStart());
  Work.clear();
  WrapperFunctionsMap.clear();

  // Emit target region as a standalone region.
  class NVPTXPrePostActionTy : public PrePostActionTy {
    CGOpenMPRuntimeNVPTX::EntryFunctionState &EST;
    CGOpenMPRuntimeNVPTX::WorkerFunctionState &WST;

  public:
    NVPTXPrePostActionTy(CGOpenMPRuntimeNVPTX::EntryFunctionState &EST,
                         CGOpenMPRuntimeNVPTX::WorkerFunctionState &WST)
        : EST(EST), WST(WST) {}
    void Enter(CodeGenFunction &CGF) override {
      static_cast<CGOpenMPRuntimeNVPTX &>(CGF.CGM.getOpenMPRuntime())
          .emitGenericEntryHeader(CGF, EST, WST);
    }
    void Exit(CodeGenFunction &CGF) override {
      static_cast<CGOpenMPRuntimeNVPTX &>(CGF.CGM.getOpenMPRuntime())
          .emitGenericEntryFooter(CGF, EST);
    }
  } Action(EST, WST);
  CodeGen.setAction(Action);
  emitTargetOutlinedFunctionHelper(D, ParentName, OutlinedFn, OutlinedFnID,
                                   IsOffloadEntry, CodeGen);

  // Now change the name of the worker function to correspond to this target
  // region's entry function.
  WST.WorkerFn->setName(OutlinedFn->getName() + "_worker");

  // Create the worker function
  emitWorkerFunction(WST);
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
  emitOutlinedFunctionCall(CGF, WST.Loc, WST.WorkerFn);
  CGF.EmitBranch(EST.ExitBB);

  CGF.EmitBlock(MasterCheckBB);
  auto *IsMaster =
      Bld.CreateICmpEQ(getNVPTXThreadID(CGF), getMasterThreadID(CGF));
  Bld.CreateCondBr(IsMaster, MasterBB, EST.ExitBB);

  CGF.EmitBlock(MasterBB);
  // First action in sequential region:
  // Initialize the state of the OpenMP runtime library on the GPU.
  // TODO: Optimize runtime initialization and pass in correct value.
  llvm::Value *Args[] = {getThreadLimit(CGF),
                         Bld.getInt16(/*RequiresOMPRuntime=*/1)};
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
  // TODO: Optimize runtime initialization and pass in correct value.
  llvm::Value *Args[] = {CGF.Builder.getInt16(/*IsOMPRuntimeInitialized=*/1)};
  CGF.EmitRuntimeCall(
      createNVPTXRuntimeFunction(OMPRTL_NVPTX__kmpc_kernel_deinit), Args);
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
  ASTContext &Ctx = CGM.getContext();

  CodeGenFunction CGF(CGM, /*suppressNewContext=*/true);
  CGF.StartFunction(GlobalDecl(), Ctx.VoidTy, WST.WorkerFn, *WST.CGFI, {},
                    WST.Loc, WST.Loc);
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

  // Set up shared arguments
  Address SharedArgs =
      CGF.CreateDefaultAlignTempAlloca(CGF.Int8PtrPtrTy, "shared_args");
  // TODO: Optimize runtime initialization and pass in correct value.
  llvm::Value *Args[] = {WorkFn.getPointer(), SharedArgs.getPointer(),
                         /*RequiresOMPRuntime=*/Bld.getInt16(1)};
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

  // Current context
  ASTContext &Ctx = CGF.getContext();

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

    // Insert call to work function via shared wrapper. The shared
    // wrapper takes exactly three arguments:
    //   - the parallelism level;
    //   - the master thread ID;
    //   - the list of references to shared arguments.
    //
    // TODO: Assert that the function is a wrapper function.s
    Address Capture = CGF.EmitLoadOfPointer(SharedArgs,
       Ctx.getPointerType(
          Ctx.getPointerType(Ctx.VoidPtrTy)).castAs<PointerType>());
    emitOutlinedFunctionCall(CGF, WST.Loc, W,
                             {Bld.getInt16(/*ParallelLevel=*/0),
                              getMasterThreadID(CGF), Capture.getPointer()});

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
    // Build void __kmpc_kernel_init(kmp_int32 thread_limit, int16_t
    // RequiresOMPRuntime);
    llvm::Type *TypeParams[] = {CGM.Int32Ty, CGM.Int16Ty};
    llvm::FunctionType *FnTy =
        llvm::FunctionType::get(CGM.VoidTy, TypeParams, /*isVarArg*/ false);
    RTLFn = CGM.CreateRuntimeFunction(FnTy, "__kmpc_kernel_init");
    break;
  }
  case OMPRTL_NVPTX__kmpc_kernel_deinit: {
    // Build void __kmpc_kernel_deinit(int16_t IsOMPRuntimeInitialized);
    llvm::Type *TypeParams[] = {CGM.Int16Ty};
    llvm::FunctionType *FnTy =
        llvm::FunctionType::get(CGM.VoidTy, TypeParams, /*isVarArg*/ false);
    RTLFn = CGM.CreateRuntimeFunction(FnTy, "__kmpc_kernel_deinit");
    break;
  }
  case OMPRTL_NVPTX__kmpc_spmd_kernel_init: {
    // Build void __kmpc_spmd_kernel_init(kmp_int32 thread_limit,
    // int16_t RequiresOMPRuntime, int16_t RequiresDataSharing);
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
    /// void *outlined_function, void ***args, kmp_int32 nArgs, int16_t
    /// IsOMPRuntimeInitialized);
    llvm::Type *TypeParams[] = {CGM.Int8PtrTy,
                                CGM.Int8PtrPtrTy->getPointerTo(0), CGM.Int32Ty,
                                CGM.Int16Ty};
    llvm::FunctionType *FnTy =
        llvm::FunctionType::get(CGM.VoidTy, TypeParams, /*isVarArg*/ false);
    RTLFn = CGM.CreateRuntimeFunction(FnTy, "__kmpc_kernel_prepare_parallel");
    break;
  }
  case OMPRTL_NVPTX__kmpc_kernel_parallel: {
    /// Build bool __kmpc_kernel_parallel(void **outlined_function, void
    /// ***args, int16_t IsOMPRuntimeInitialized);
    llvm::Type *TypeParams[] = {CGM.Int8PtrPtrTy,
                                CGM.Int8PtrPtrTy->getPointerTo(0), CGM.Int16Ty};
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
  case OMPRTL_NVPTX__kmpc_shuffle_int32: {
    // Build int32_t __kmpc_shuffle_int32(int32_t element,
    // int16_t lane_offset, int16_t warp_size);
    llvm::Type *TypeParams[] = {CGM.Int32Ty, CGM.Int16Ty, CGM.Int16Ty};
    llvm::FunctionType *FnTy =
        llvm::FunctionType::get(CGM.Int32Ty, TypeParams, /*isVarArg*/ false);
    RTLFn = CGM.CreateRuntimeFunction(FnTy, "__kmpc_shuffle_int32");
    break;
  }
  case OMPRTL_NVPTX__kmpc_shuffle_int64: {
    // Build int64_t __kmpc_shuffle_int64(int64_t element,
    // int16_t lane_offset, int16_t warp_size);
    llvm::Type *TypeParams[] = {CGM.Int64Ty, CGM.Int16Ty, CGM.Int16Ty};
    llvm::FunctionType *FnTy =
        llvm::FunctionType::get(CGM.Int64Ty, TypeParams, /*isVarArg*/ false);
    RTLFn = CGM.CreateRuntimeFunction(FnTy, "__kmpc_shuffle_int64");
    break;
  }
  case OMPRTL_NVPTX__kmpc_parallel_reduce_nowait: {
    // Build int32_t kmpc_nvptx_parallel_reduce_nowait(kmp_int32 global_tid,
    // kmp_int32 num_vars, size_t reduce_size, void* reduce_data,
    // void (*kmp_ShuffleReductFctPtr)(void *rhsData, int16_t lane_id, int16_t
    // lane_offset, int16_t Algorithm Version),
    // void (*kmp_InterWarpCopyFctPtr)(void* src, int warp_num));
    llvm::Type *ShuffleReduceTypeParams[] = {CGM.VoidPtrTy, CGM.Int16Ty,
                                             CGM.Int16Ty, CGM.Int16Ty};
    auto *ShuffleReduceFnTy =
        llvm::FunctionType::get(CGM.VoidTy, ShuffleReduceTypeParams,
                                /*isVarArg=*/false);
    llvm::Type *InterWarpCopyTypeParams[] = {CGM.VoidPtrTy, CGM.Int32Ty};
    auto *InterWarpCopyFnTy =
        llvm::FunctionType::get(CGM.VoidTy, InterWarpCopyTypeParams,
                                /*isVarArg=*/false);
    llvm::Type *TypeParams[] = {CGM.Int32Ty,
                                CGM.Int32Ty,
                                CGM.SizeTy,
                                CGM.VoidPtrTy,
                                ShuffleReduceFnTy->getPointerTo(),
                                InterWarpCopyFnTy->getPointerTo()};
    llvm::FunctionType *FnTy =
        llvm::FunctionType::get(CGM.Int32Ty, TypeParams, /*isVarArg=*/false);
    RTLFn = CGM.CreateRuntimeFunction(
        FnTy, /*Name=*/"__kmpc_nvptx_parallel_reduce_nowait");
    break;
  }
  case OMPRTL_NVPTX__kmpc_teams_reduce_nowait: {
    // Build int32_t __kmpc_nvptx_teams_reduce_nowait(int32_t global_tid,
    // int32_t num_vars, size_t reduce_size, void *reduce_data,
    // void (*kmp_ShuffleReductFctPtr)(void *rhsData, int16_t lane_id, int16_t
    // lane_offset, int16_t shortCircuit),
    // void (*kmp_InterWarpCopyFctPtr)(void* src, int32_t warp_num),
    // void (*kmp_CopyToScratchpadFctPtr)(void *reduce_data, void * scratchpad,
    // int32_t index, int32_t width),
    // void (*kmp_LoadReduceFctPtr)(void *reduce_data, void * scratchpad,
    // int32_t index, int32_t width, int32_t reduce))
    llvm::Type *ShuffleReduceTypeParams[] = {CGM.VoidPtrTy, CGM.Int16Ty,
                                             CGM.Int16Ty, CGM.Int16Ty};
    auto *ShuffleReduceFnTy =
        llvm::FunctionType::get(CGM.VoidTy, ShuffleReduceTypeParams,
                                /*isVarArg=*/false);
    llvm::Type *InterWarpCopyTypeParams[] = {CGM.VoidPtrTy, CGM.Int32Ty};
    auto *InterWarpCopyFnTy =
        llvm::FunctionType::get(CGM.VoidTy, InterWarpCopyTypeParams,
                                /*isVarArg=*/false);
    llvm::Type *CopyToScratchpadTypeParams[] = {CGM.VoidPtrTy, CGM.VoidPtrTy,
                                                CGM.Int32Ty, CGM.Int32Ty};
    auto *CopyToScratchpadFnTy =
        llvm::FunctionType::get(CGM.VoidTy, CopyToScratchpadTypeParams,
                                /*isVarArg=*/false);
    llvm::Type *LoadReduceTypeParams[] = {
        CGM.VoidPtrTy, CGM.VoidPtrTy, CGM.Int32Ty, CGM.Int32Ty, CGM.Int32Ty};
    auto *LoadReduceFnTy =
        llvm::FunctionType::get(CGM.VoidTy, LoadReduceTypeParams,
                                /*isVarArg=*/false);
    llvm::Type *TypeParams[] = {CGM.Int32Ty,
                                CGM.Int32Ty,
                                CGM.SizeTy,
                                CGM.VoidPtrTy,
                                ShuffleReduceFnTy->getPointerTo(),
                                InterWarpCopyFnTy->getPointerTo(),
                                CopyToScratchpadFnTy->getPointerTo(),
                                LoadReduceFnTy->getPointerTo()};
    llvm::FunctionType *FnTy =
        llvm::FunctionType::get(CGM.Int32Ty, TypeParams, /*isVarArg=*/false);
    RTLFn = CGM.CreateRuntimeFunction(
        FnTy, /*Name=*/"__kmpc_nvptx_teams_reduce_nowait");
    break;
  }
  case OMPRTL_NVPTX__kmpc_end_reduce_nowait: {
    // Build __kmpc_end_reduce_nowait(kmp_int32 global_tid);
    llvm::Type *TypeParams[] = {CGM.Int32Ty};
    llvm::FunctionType *FnTy =
        llvm::FunctionType::get(CGM.VoidTy, TypeParams, /*isVarArg=*/false);
    RTLFn = CGM.CreateRuntimeFunction(
        FnTy, /*Name=*/"__kmpc_nvptx_end_reduce_nowait");
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

  auto *OutlinedFun = cast<llvm::Function>(
    CGOpenMPRuntime::emitParallelOutlinedFunction(
          D, ThreadIDVar, InnermostKind, CodeGen));
  if (!isInSpmdExecutionMode()) {
    llvm::Function *WrapperFun =
        createDataSharingWrapper(OutlinedFun, D);
    WrapperFunctionsMap[OutlinedFun] = WrapperFun;
  }

  return OutlinedFun;
}

llvm::Value *CGOpenMPRuntimeNVPTX::emitTeamsOutlinedFunction(
    const OMPExecutableDirective &D, const VarDecl *ThreadIDVar,
    OpenMPDirectiveKind InnermostKind, const RegionCodeGenTy &CodeGen) {

  llvm::Value *OutlinedFunVal = CGOpenMPRuntime::emitTeamsOutlinedFunction(
      D, ThreadIDVar, InnermostKind, CodeGen);
  llvm::Function *OutlinedFun = cast<llvm::Function>(OutlinedFunVal);
  OutlinedFun->removeFnAttr(llvm::Attribute::NoInline);
  OutlinedFun->removeFnAttr(llvm::Attribute::OptimizeNone);
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
  emitOutlinedFunctionCall(CGF, Loc, OutlinedFn, OutlinedFnArgs);
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
  llvm::Function *WFn = WrapperFunctionsMap[Fn];
  assert(WFn && "Wrapper function does not exist!");

  // Force inline this outlined function at its call site.
  Fn->setLinkage(llvm::GlobalValue::InternalLinkage);

  auto &&L0ParallelGen = [this, WFn, &CapturedVars](CodeGenFunction &CGF,
                                                    PrePostActionTy &) {
    CGBuilderTy &Bld = CGF.Builder;

    llvm::Value *ID = Bld.CreateBitOrPointerCast(WFn, CGM.Int8PtrTy);

    if (!CapturedVars.empty()) {
      // There's somehting to share, add the attribute
      CGF.CurFn->addFnAttr("has-nvptx-shared-depot");
      // Prepare for parallel region. Indicate the outlined function.
      Address SharedArgs =
          CGF.CreateDefaultAlignTempAlloca(CGF.VoidPtrPtrTy,
              "shared_args");
      llvm::Value *SharedArgsPtr = SharedArgs.getPointer();
      // TODO: Optimize runtime initialization and pass in correct value.
      llvm::Value *Args[] = {ID, SharedArgsPtr,
                             Bld.getInt32(CapturedVars.size()),
                             /*RequiresOMPRuntime=*/Bld.getInt16(1)};

      CGF.EmitRuntimeCall(
          createNVPTXRuntimeFunction(OMPRTL_NVPTX__kmpc_kernel_prepare_parallel),
          Args);

      unsigned Idx = 0;
      ASTContext &Ctx = CGF.getContext();
      for (llvm::Value *V : CapturedVars) {
        Address Dst = Bld.CreateConstInBoundsGEP(
            CGF.EmitLoadOfPointer(SharedArgs,
            Ctx.getPointerType(
                Ctx.getPointerType(Ctx.VoidPtrTy)).castAs<PointerType>()),
            Idx, CGF.getPointerSize());
        llvm::Value *PtrV = Bld.CreateBitCast(V, CGF.VoidPtrTy);
        CGF.EmitStoreOfScalar(PtrV, Dst, /*Volatile=*/false,
            Ctx.getPointerType(Ctx.VoidPtrTy));
        Idx++;
      }
    } else {
      // TODO: Optimize runtime initialization and pass in correct value.
      llvm::Value *Args[] = {
          ID, llvm::ConstantPointerNull::get(CGF.VoidPtrPtrTy->getPointerTo(0)),
          /*nArgs=*/Bld.getInt32(0), /*RequiresOMPRuntime=*/Bld.getInt16(1)};
      CGF.EmitRuntimeCall(
          createNVPTXRuntimeFunction(OMPRTL_NVPTX__kmpc_kernel_prepare_parallel),
          Args);
    }

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
    Work.emplace_back(WFn);
  };

  auto *RTLoc = emitUpdateLocation(CGF, Loc);
  auto *ThreadID = getThreadID(CGF, Loc);
  llvm::Value *Args[] = {RTLoc, ThreadID};

  auto &&SeqGen = [this, Fn, &CapturedVars, &Args, Loc](CodeGenFunction &CGF,
                                                        PrePostActionTy &) {
    auto &&CodeGen = [this, Fn, &CapturedVars, Loc](CodeGenFunction &CGF,
                                                    PrePostActionTy &Action) {
      Action.Enter(CGF);

      llvm::SmallVector<llvm::Value *, 16> OutlinedFnArgs;
      OutlinedFnArgs.push_back(
          llvm::ConstantPointerNull::get(CGM.Int32Ty->getPointerTo()));
      OutlinedFnArgs.push_back(
          llvm::ConstantPointerNull::get(CGM.Int32Ty->getPointerTo()));
      OutlinedFnArgs.append(CapturedVars.begin(), CapturedVars.end());
      emitOutlinedFunctionCall(CGF, Loc, Fn, OutlinedFnArgs);
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
  emitOutlinedFunctionCall(CGF, Loc, OutlinedFn, OutlinedFnArgs);
}

/// Cast value to the specified type.
static llvm::Value *
castValueToType(CodeGenFunction &CGF, llvm::Value *Val, llvm::Type *CastTy,
                llvm::Optional<bool> IsSigned = llvm::None) {
  if (Val->getType() == CastTy)
    return Val;
  if (Val->getType()->getPrimitiveSizeInBits() > 0 &&
      CastTy->getPrimitiveSizeInBits() > 0 &&
      Val->getType()->getPrimitiveSizeInBits() ==
          CastTy->getPrimitiveSizeInBits())
    return CGF.Builder.CreateBitCast(Val, CastTy);
  if (IsSigned.hasValue() && CastTy->isIntegerTy() &&
      Val->getType()->isIntegerTy())
    return CGF.Builder.CreateIntCast(Val, CastTy, *IsSigned);
  Address CastItem = CGF.CreateTempAlloca(
      CastTy,
      CharUnits::fromQuantity(
          CGF.CGM.getDataLayout().getPrefTypeAlignment(Val->getType())));
  Address ValCastItem = CGF.Builder.CreatePointerBitCastOrAddrSpaceCast(
      CastItem, Val->getType()->getPointerTo(CastItem.getAddressSpace()));
  CGF.Builder.CreateStore(Val, ValCastItem);
  return CGF.Builder.CreateLoad(CastItem);
}

/// This function creates calls to one of two shuffle functions to copy
/// variables between lanes in a warp.
static llvm::Value *createRuntimeShuffleFunction(CodeGenFunction &CGF,
                                                 llvm::Value *Elem,
                                                 llvm::Value *Offset) {
  auto &CGM = CGF.CGM;
  auto &Bld = CGF.Builder;
  CGOpenMPRuntimeNVPTX &RT =
      *(static_cast<CGOpenMPRuntimeNVPTX *>(&CGM.getOpenMPRuntime()));

  unsigned Size = CGM.getDataLayout().getTypeStoreSize(Elem->getType());
  assert(Size <= 8 && "Unsupported bitwidth in shuffle instruction.");

  OpenMPRTLFunctionNVPTX ShuffleFn = Size <= 4
                                         ? OMPRTL_NVPTX__kmpc_shuffle_int32
                                         : OMPRTL_NVPTX__kmpc_shuffle_int64;

  // Cast all types to 32- or 64-bit values before calling shuffle routines.
  llvm::Type *CastTy = Size <= 4 ? CGM.Int32Ty : CGM.Int64Ty;
  llvm::Value *ElemCast = castValueToType(CGF, Elem, CastTy, /*isSigned=*/true);
  auto *WarpSize =
      Bld.CreateIntCast(getNVPTXWarpSize(CGF), CGM.Int16Ty, /*isSigned=*/true);

  auto *ShuffledVal =
      CGF.EmitRuntimeCall(RT.createNVPTXRuntimeFunction(ShuffleFn),
                          {ElemCast, Offset, WarpSize});

  return castValueToType(CGF, ShuffledVal, Elem->getType(), /*isSigned=*/true);
}

namespace {
enum CopyAction : unsigned {
  // RemoteLaneToThread: Copy over a Reduce list from a remote lane in
  // the warp using shuffle instructions.
  RemoteLaneToThread,
  // ThreadCopy: Make a copy of a Reduce list on the thread's stack.
  ThreadCopy,
  // ThreadToScratchpad: Copy a team-reduced array to the scratchpad.
  ThreadToScratchpad,
  // ScratchpadToThread: Copy from a scratchpad array in global memory
  // containing team-reduced data to a thread's stack.
  ScratchpadToThread,
};
} // namespace

struct CopyOptionsTy {
  llvm::Value *RemoteLaneOffset;
  llvm::Value *ScratchpadIndex;
  llvm::Value *ScratchpadWidth;
};

/// Emit instructions to copy a Reduce list, which contains partially
/// aggregated values, in the specified direction.
static void emitReductionListCopy(
    CopyAction Action, CodeGenFunction &CGF, QualType ReductionArrayTy,
    ArrayRef<const Expr *> Privates, Address SrcBase, Address DestBase,
    CopyOptionsTy CopyOptions = {nullptr, nullptr, nullptr}) {

  auto &CGM = CGF.CGM;
  auto &C = CGM.getContext();
  auto &Bld = CGF.Builder;

  auto *RemoteLaneOffset = CopyOptions.RemoteLaneOffset;
  auto *ScratchpadIndex = CopyOptions.ScratchpadIndex;
  auto *ScratchpadWidth = CopyOptions.ScratchpadWidth;

  // Iterates, element-by-element, through the source Reduce list and
  // make a copy.
  unsigned Idx = 0;
  unsigned Size = Privates.size();
  for (auto &Private : Privates) {
    Address SrcElementAddr = Address::invalid();
    Address DestElementAddr = Address::invalid();
    Address DestElementPtrAddr = Address::invalid();
    // Should we shuffle in an element from a remote lane?
    bool ShuffleInElement = false;
    // Set to true to update the pointer in the dest Reduce list to a
    // newly created element.
    bool UpdateDestListPtr = false;
    // Increment the src or dest pointer to the scratchpad, for each
    // new element.
    bool IncrScratchpadSrc = false;
    bool IncrScratchpadDest = false;

    switch (Action) {
    case RemoteLaneToThread: {
      // Step 1.1: Get the address for the src element in the Reduce list.
      Address SrcElementPtrAddr =
          Bld.CreateConstArrayGEP(SrcBase, Idx, CGF.getPointerSize());
      SrcElementAddr = CGF.EmitLoadOfPointer(
          SrcElementPtrAddr,
          C.getPointerType(Private->getType())->castAs<PointerType>());

      // Step 1.2: Create a temporary to store the element in the destination
      // Reduce list.
      DestElementPtrAddr =
          Bld.CreateConstArrayGEP(DestBase, Idx, CGF.getPointerSize());
      DestElementAddr =
          CGF.CreateMemTemp(Private->getType(), ".omp.reduction.element");
      ShuffleInElement = true;
      UpdateDestListPtr = true;
      break;
    }
    case ThreadCopy: {
      // Step 1.1: Get the address for the src element in the Reduce list.
      Address SrcElementPtrAddr =
          Bld.CreateConstArrayGEP(SrcBase, Idx, CGF.getPointerSize());
      SrcElementAddr = CGF.EmitLoadOfPointer(
          SrcElementPtrAddr,
          C.getPointerType(Private->getType())->castAs<PointerType>());

      // Step 1.2: Get the address for dest element.  The destination
      // element has already been created on the thread's stack.
      DestElementPtrAddr =
          Bld.CreateConstArrayGEP(DestBase, Idx, CGF.getPointerSize());
      DestElementAddr = CGF.EmitLoadOfPointer(
          DestElementPtrAddr,
          C.getPointerType(Private->getType())->castAs<PointerType>());
      break;
    }
    case ThreadToScratchpad: {
      // Step 1.1: Get the address for the src element in the Reduce list.
      Address SrcElementPtrAddr =
          Bld.CreateConstArrayGEP(SrcBase, Idx, CGF.getPointerSize());
      SrcElementAddr = CGF.EmitLoadOfPointer(
          SrcElementPtrAddr,
          C.getPointerType(Private->getType())->castAs<PointerType>());

      // Step 1.2: Get the address for dest element:
      // address = base + index * ElementSizeInChars.
      unsigned ElementSizeInChars =
          C.getTypeSizeInChars(Private->getType()).getQuantity();
      auto *CurrentOffset =
          Bld.CreateMul(llvm::ConstantInt::get(CGM.SizeTy, ElementSizeInChars),
                        ScratchpadIndex);
      auto *ScratchPadElemAbsolutePtrVal =
          Bld.CreateAdd(DestBase.getPointer(), CurrentOffset);
      ScratchPadElemAbsolutePtrVal =
          Bld.CreateIntToPtr(ScratchPadElemAbsolutePtrVal, CGF.VoidPtrTy);
      DestElementAddr = Address(ScratchPadElemAbsolutePtrVal,
                                C.getTypeAlignInChars(Private->getType()));
      IncrScratchpadDest = true;
      break;
    }
    case ScratchpadToThread: {
      // Step 1.1: Get the address for the src element in the scratchpad.
      // address = base + index * ElementSizeInChars.
      unsigned ElementSizeInChars =
          C.getTypeSizeInChars(Private->getType()).getQuantity();
      auto *CurrentOffset =
          Bld.CreateMul(llvm::ConstantInt::get(CGM.SizeTy, ElementSizeInChars),
                        ScratchpadIndex);
      auto *ScratchPadElemAbsolutePtrVal =
          Bld.CreateAdd(SrcBase.getPointer(), CurrentOffset);
      ScratchPadElemAbsolutePtrVal =
          Bld.CreateIntToPtr(ScratchPadElemAbsolutePtrVal, CGF.VoidPtrTy);
      SrcElementAddr = Address(ScratchPadElemAbsolutePtrVal,
                               C.getTypeAlignInChars(Private->getType()));
      IncrScratchpadSrc = true;

      // Step 1.2: Create a temporary to store the element in the destination
      // Reduce list.
      DestElementPtrAddr =
          Bld.CreateConstArrayGEP(DestBase, Idx, CGF.getPointerSize());
      DestElementAddr =
          CGF.CreateMemTemp(Private->getType(), ".omp.reduction.element");
      UpdateDestListPtr = true;
      break;
    }
    }

    // Regardless of src and dest of copy, we emit the load of src
    // element as this is required in all directions
    SrcElementAddr = Bld.CreateElementBitCast(
        SrcElementAddr, CGF.ConvertTypeForMem(Private->getType()));
    llvm::Value *Elem =
        CGF.EmitLoadOfScalar(SrcElementAddr, /*Volatile=*/false,
                             Private->getType(), SourceLocation());

    // Now that all active lanes have read the element in the
    // Reduce list, shuffle over the value from the remote lane.
    if (ShuffleInElement)
      Elem = createRuntimeShuffleFunction(CGF, Elem, RemoteLaneOffset);

    DestElementAddr = Bld.CreateElementBitCast(DestElementAddr,
                                               SrcElementAddr.getElementType());

    // Store the source element value to the dest element address.
    CGF.EmitStoreOfScalar(Elem, DestElementAddr, /*Volatile=*/false,
                          Private->getType());

    // Step 3.1: Modify reference in dest Reduce list as needed.
    // Modifying the reference in Reduce list to point to the newly
    // created element.  The element is live in the current function
    // scope and that of functions it invokes (i.e., reduce_function).
    // RemoteReduceData[i] = (void*)&RemoteElem
    if (UpdateDestListPtr) {
      CGF.EmitStoreOfScalar(Bld.CreatePointerBitCastOrAddrSpaceCast(
                                DestElementAddr.getPointer(), CGF.VoidPtrTy),
                            DestElementPtrAddr, /*Volatile=*/false,
                            C.VoidPtrTy);
    }

    // Step 4.1: Increment SrcBase/DestBase so that it points to the starting
    // address of the next element in scratchpad memory, unless we're currently
    // processing the last one.  Memory alignment is also taken care of here.
    if ((IncrScratchpadDest || IncrScratchpadSrc) && (Idx + 1 < Size)) {
      llvm::Value *ScratchpadBasePtr =
          IncrScratchpadDest ? DestBase.getPointer() : SrcBase.getPointer();
      unsigned ElementSizeInChars =
          C.getTypeSizeInChars(Private->getType()).getQuantity();
      ScratchpadBasePtr = Bld.CreateAdd(
          ScratchpadBasePtr,
          Bld.CreateMul(ScratchpadWidth, llvm::ConstantInt::get(
                                             CGM.SizeTy, ElementSizeInChars)));

      // Take care of global memory alignment for performance
      ScratchpadBasePtr = Bld.CreateSub(ScratchpadBasePtr,
                                        llvm::ConstantInt::get(CGM.SizeTy, 1));
      ScratchpadBasePtr = Bld.CreateSDiv(
          ScratchpadBasePtr,
          llvm::ConstantInt::get(CGM.SizeTy, GlobalMemoryAlignment));
      ScratchpadBasePtr = Bld.CreateAdd(ScratchpadBasePtr,
                                        llvm::ConstantInt::get(CGM.SizeTy, 1));
      ScratchpadBasePtr = Bld.CreateMul(
          ScratchpadBasePtr,
          llvm::ConstantInt::get(CGM.SizeTy, GlobalMemoryAlignment));

      if (IncrScratchpadDest)
        DestBase = Address(ScratchpadBasePtr, CGF.getPointerAlign());
      else /* IncrScratchpadSrc = true */
        SrcBase = Address(ScratchpadBasePtr, CGF.getPointerAlign());
    }

    Idx++;
  }
}

/// This function emits a helper that loads data from the scratchpad array
/// and (optionally) reduces it with the input operand.
///
///  load_and_reduce(local, scratchpad, index, width, should_reduce)
///  reduce_data remote;
///  for elem in remote:
///    remote.elem = Scratchpad[elem_id][index]
///  if (should_reduce)
///    local = local @ remote
///  else
///    local = remote
static llvm::Value *emitReduceScratchpadFunction(
    CodeGenModule &CGM, ArrayRef<const Expr *> Privates,
    QualType ReductionArrayTy, llvm::Value *ReduceFn, SourceLocation Loc) {
  auto &C = CGM.getContext();
  auto Int32Ty = C.getIntTypeForBitwidth(32, /* Signed */ true);

  // Destination of the copy.
  ImplicitParamDecl ReduceListArg(C, /*DC=*/nullptr, Loc, /*Id=*/nullptr,
                                  C.VoidPtrTy, ImplicitParamDecl::Other);
  // Base address of the scratchpad array, with each element storing a
  // Reduce list per team.
  ImplicitParamDecl ScratchPadArg(C, /*DC=*/nullptr, Loc, /*Id=*/nullptr,
                                  C.VoidPtrTy, ImplicitParamDecl::Other);
  // A source index into the scratchpad array.
  ImplicitParamDecl IndexArg(C, /*DC=*/nullptr, Loc, /*Id=*/nullptr, Int32Ty,
                             ImplicitParamDecl::Other);
  // Row width of an element in the scratchpad array, typically
  // the number of teams.
  ImplicitParamDecl WidthArg(C, /*DC=*/nullptr, Loc, /*Id=*/nullptr, Int32Ty,
                             ImplicitParamDecl::Other);
  // If should_reduce == 1, then it's load AND reduce,
  // If should_reduce == 0 (or otherwise), then it only loads (+ copy).
  // The latter case is used for initialization.
  ImplicitParamDecl ShouldReduceArg(C, /*DC=*/nullptr, Loc, /*Id=*/nullptr,
                                    Int32Ty, ImplicitParamDecl::Other);

  FunctionArgList Args;
  Args.push_back(&ReduceListArg);
  Args.push_back(&ScratchPadArg);
  Args.push_back(&IndexArg);
  Args.push_back(&WidthArg);
  Args.push_back(&ShouldReduceArg);

  auto &CGFI = CGM.getTypes().arrangeBuiltinFunctionDeclaration(C.VoidTy, Args);
  auto *Fn = llvm::Function::Create(
      CGM.getTypes().GetFunctionType(CGFI), llvm::GlobalValue::InternalLinkage,
      "_omp_reduction_load_and_reduce", &CGM.getModule());
  CGM.SetInternalFunctionAttributes(/*DC=*/nullptr, Fn, CGFI);
  CodeGenFunction CGF(CGM);
  CGF.StartFunction(GlobalDecl(), C.VoidTy, Fn, CGFI, Args, Loc, Loc);

  auto &Bld = CGF.Builder;

  // Get local Reduce list pointer.
  Address AddrReduceListArg = CGF.GetAddrOfLocalVar(&ReduceListArg);
  Address ReduceListAddr(
      Bld.CreatePointerBitCastOrAddrSpaceCast(
          CGF.EmitLoadOfScalar(AddrReduceListArg, /*Volatile=*/false,
                               C.VoidPtrTy, SourceLocation()),
          CGF.ConvertTypeForMem(ReductionArrayTy)->getPointerTo()),
      CGF.getPointerAlign());

  Address AddrScratchPadArg = CGF.GetAddrOfLocalVar(&ScratchPadArg);
  llvm::Value *ScratchPadBase = CGF.EmitLoadOfScalar(
      AddrScratchPadArg, /*Volatile=*/false, C.VoidPtrTy, SourceLocation());

  Address AddrIndexArg = CGF.GetAddrOfLocalVar(&IndexArg);
  llvm::Value *IndexVal =
      Bld.CreateIntCast(CGF.EmitLoadOfScalar(AddrIndexArg, /*Volatile=*/false,
                                             Int32Ty, SourceLocation()),
                        CGM.SizeTy, /*isSigned=*/true);

  Address AddrWidthArg = CGF.GetAddrOfLocalVar(&WidthArg);
  llvm::Value *WidthVal =
      Bld.CreateIntCast(CGF.EmitLoadOfScalar(AddrWidthArg, /*Volatile=*/false,
                                             Int32Ty, SourceLocation()),
                        CGM.SizeTy, /*isSigned=*/true);

  Address AddrShouldReduceArg = CGF.GetAddrOfLocalVar(&ShouldReduceArg);
  llvm::Value *ShouldReduceVal = CGF.EmitLoadOfScalar(
      AddrShouldReduceArg, /*Volatile=*/false, Int32Ty, SourceLocation());

  // The absolute ptr address to the base addr of the next element to copy.
  llvm::Value *CumulativeElemBasePtr =
      Bld.CreatePtrToInt(ScratchPadBase, CGM.SizeTy);
  Address SrcDataAddr(CumulativeElemBasePtr, CGF.getPointerAlign());

  // Create a Remote Reduce list to store the elements read from the
  // scratchpad array.
  Address RemoteReduceList =
      CGF.CreateMemTemp(ReductionArrayTy, ".omp.reduction.remote_red_list");

  // Assemble remote Reduce list from scratchpad array.
  emitReductionListCopy(ScratchpadToThread, CGF, ReductionArrayTy, Privates,
                        SrcDataAddr, RemoteReduceList,
                        {/*RemoteLaneOffset=*/nullptr,
                         /*ScratchpadIndex=*/IndexVal,
                         /*ScratchpadWidth=*/WidthVal});

  llvm::BasicBlock *ThenBB = CGF.createBasicBlock("then");
  llvm::BasicBlock *ElseBB = CGF.createBasicBlock("else");
  llvm::BasicBlock *MergeBB = CGF.createBasicBlock("ifcont");

  auto CondReduce = Bld.CreateICmpEQ(ShouldReduceVal, Bld.getInt32(1));
  Bld.CreateCondBr(CondReduce, ThenBB, ElseBB);

  CGF.EmitBlock(ThenBB);
  // We should reduce with the local Reduce list.
  // reduce_function(LocalReduceList, RemoteReduceList)
  llvm::Value *LocalDataPtr = Bld.CreatePointerBitCastOrAddrSpaceCast(
      ReduceListAddr.getPointer(), CGF.VoidPtrTy);
  llvm::Value *RemoteDataPtr = Bld.CreatePointerBitCastOrAddrSpaceCast(
      RemoteReduceList.getPointer(), CGF.VoidPtrTy);
  CGM.getOpenMPRuntime().emitOutlinedFunctionCall(
      CGF, Loc, ReduceFn, {LocalDataPtr, RemoteDataPtr});
  Bld.CreateBr(MergeBB);

  CGF.EmitBlock(ElseBB);
  // No reduction; just copy:
  // Local Reduce list = Remote Reduce list.
  emitReductionListCopy(ThreadCopy, CGF, ReductionArrayTy, Privates,
                        RemoteReduceList, ReduceListAddr);
  Bld.CreateBr(MergeBB);

  CGF.EmitBlock(MergeBB);

  CGF.FinishFunction();
  return Fn;
}

/// This function emits a helper that stores reduced data from the team
/// master to a scratchpad array in global memory.
///
///  for elem in Reduce List:
///    scratchpad[elem_id][index] = elem
///
static llvm::Value *emitCopyToScratchpad(CodeGenModule &CGM,
                                         ArrayRef<const Expr *> Privates,
                                         QualType ReductionArrayTy,
                                         SourceLocation Loc) {

  auto &C = CGM.getContext();
  auto Int32Ty = C.getIntTypeForBitwidth(32, /* Signed */ true);

  // Source of the copy.
  ImplicitParamDecl ReduceListArg(C, /*DC=*/nullptr, Loc, /*Id=*/nullptr,
                                  C.VoidPtrTy, ImplicitParamDecl::Other);
  // Base address of the scratchpad array, with each element storing a
  // Reduce list per team.
  ImplicitParamDecl ScratchPadArg(C, /*DC=*/nullptr, Loc, /*Id=*/nullptr,
                                  C.VoidPtrTy, ImplicitParamDecl::Other);
  // A destination index into the scratchpad array, typically the team
  // identifier.
  ImplicitParamDecl IndexArg(C, /*DC=*/nullptr, Loc, /*Id=*/nullptr, Int32Ty,
                             ImplicitParamDecl::Other);
  // Row width of an element in the scratchpad array, typically
  // the number of teams.
  ImplicitParamDecl WidthArg(C, /*DC=*/nullptr, Loc, /*Id=*/nullptr, Int32Ty,
                             ImplicitParamDecl::Other);

  FunctionArgList Args;
  Args.push_back(&ReduceListArg);
  Args.push_back(&ScratchPadArg);
  Args.push_back(&IndexArg);
  Args.push_back(&WidthArg);

  auto &CGFI = CGM.getTypes().arrangeBuiltinFunctionDeclaration(C.VoidTy, Args);
  auto *Fn = llvm::Function::Create(
      CGM.getTypes().GetFunctionType(CGFI), llvm::GlobalValue::InternalLinkage,
      "_omp_reduction_copy_to_scratchpad", &CGM.getModule());
  CGM.SetInternalFunctionAttributes(/*DC=*/nullptr, Fn, CGFI);
  CodeGenFunction CGF(CGM);
  CGF.StartFunction(GlobalDecl(), C.VoidTy, Fn, CGFI, Args, Loc, Loc);

  auto &Bld = CGF.Builder;

  Address AddrReduceListArg = CGF.GetAddrOfLocalVar(&ReduceListArg);
  Address SrcDataAddr(
      Bld.CreatePointerBitCastOrAddrSpaceCast(
          CGF.EmitLoadOfScalar(AddrReduceListArg, /*Volatile=*/false,
                               C.VoidPtrTy, SourceLocation()),
          CGF.ConvertTypeForMem(ReductionArrayTy)->getPointerTo()),
      CGF.getPointerAlign());

  Address AddrScratchPadArg = CGF.GetAddrOfLocalVar(&ScratchPadArg);
  llvm::Value *ScratchPadBase = CGF.EmitLoadOfScalar(
      AddrScratchPadArg, /*Volatile=*/false, C.VoidPtrTy, SourceLocation());

  Address AddrIndexArg = CGF.GetAddrOfLocalVar(&IndexArg);
  llvm::Value *IndexVal =
      Bld.CreateIntCast(CGF.EmitLoadOfScalar(AddrIndexArg, /*Volatile=*/false,
                                             Int32Ty, SourceLocation()),
                        CGF.SizeTy, /*isSigned=*/true);

  Address AddrWidthArg = CGF.GetAddrOfLocalVar(&WidthArg);
  llvm::Value *WidthVal =
      Bld.CreateIntCast(CGF.EmitLoadOfScalar(AddrWidthArg, /*Volatile=*/false,
                                             Int32Ty, SourceLocation()),
                        CGF.SizeTy, /*isSigned=*/true);

  // The absolute ptr address to the base addr of the next element to copy.
  llvm::Value *CumulativeElemBasePtr =
      Bld.CreatePtrToInt(ScratchPadBase, CGM.SizeTy);
  Address DestDataAddr(CumulativeElemBasePtr, CGF.getPointerAlign());

  emitReductionListCopy(ThreadToScratchpad, CGF, ReductionArrayTy, Privates,
                        SrcDataAddr, DestDataAddr,
                        {/*RemoteLaneOffset=*/nullptr,
                         /*ScratchpadIndex=*/IndexVal,
                         /*ScratchpadWidth=*/WidthVal});

  CGF.FinishFunction();
  return Fn;
}

/// This function emits a helper that gathers Reduce lists from the first
/// lane of every active warp to lanes in the first warp.
///
/// void inter_warp_copy_func(void* reduce_data, num_warps)
///   shared smem[warp_size];
///   For all data entries D in reduce_data:
///     If (I am the first lane in each warp)
///       Copy my local D to smem[warp_id]
///     sync
///     if (I am the first warp)
///       Copy smem[thread_id] to my local D
///     sync
static llvm::Value *emitInterWarpCopyFunction(CodeGenModule &CGM,
                                              ArrayRef<const Expr *> Privates,
                                              QualType ReductionArrayTy,
                                              SourceLocation Loc) {
  auto &C = CGM.getContext();
  auto &M = CGM.getModule();

  // ReduceList: thread local Reduce list.
  // At the stage of the computation when this function is called, partially
  // aggregated values reside in the first lane of every active warp.
  ImplicitParamDecl ReduceListArg(C, /*DC=*/nullptr, Loc, /*Id=*/nullptr,
                                  C.VoidPtrTy, ImplicitParamDecl::Other);
  // NumWarps: number of warps active in the parallel region.  This could
  // be smaller than 32 (max warps in a CTA) for partial block reduction.
  ImplicitParamDecl NumWarpsArg(C, /*DC=*/nullptr, Loc, /*Id=*/nullptr,
                                C.getIntTypeForBitwidth(32, /* Signed */ true),
                                ImplicitParamDecl::Other);
  FunctionArgList Args;
  Args.push_back(&ReduceListArg);
  Args.push_back(&NumWarpsArg);

  auto &CGFI = CGM.getTypes().arrangeBuiltinFunctionDeclaration(C.VoidTy, Args);
  auto *Fn = llvm::Function::Create(
      CGM.getTypes().GetFunctionType(CGFI), llvm::GlobalValue::InternalLinkage,
      "_omp_reduction_inter_warp_copy_func", &CGM.getModule());
  CGM.SetInternalFunctionAttributes(/*DC=*/nullptr, Fn, CGFI);
  CodeGenFunction CGF(CGM);
  CGF.StartFunction(GlobalDecl(), C.VoidTy, Fn, CGFI, Args, Loc, Loc);

  auto &Bld = CGF.Builder;

  // This array is used as a medium to transfer, one reduce element at a time,
  // the data from the first lane of every warp to lanes in the first warp
  // in order to perform the final step of a reduction in a parallel region
  // (reduction across warps).  The array is placed in NVPTX __shared__ memory
  // for reduced latency, as well as to have a distinct copy for concurrently
  // executing target regions.  The array is declared with common linkage so
  // as to be shared across compilation units.
  const char *TransferMediumName =
      "__openmp_nvptx_data_transfer_temporary_storage";
  llvm::GlobalVariable *TransferMedium =
      M.getGlobalVariable(TransferMediumName);
  if (!TransferMedium) {
    auto *Ty = llvm::ArrayType::get(CGM.Int64Ty, WarpSize);
    unsigned SharedAddressSpace = C.getTargetAddressSpace(LangAS::cuda_shared);
    TransferMedium = new llvm::GlobalVariable(
        M, Ty,
        /*isConstant=*/false, llvm::GlobalVariable::CommonLinkage,
        llvm::Constant::getNullValue(Ty), TransferMediumName,
        /*InsertBefore=*/nullptr, llvm::GlobalVariable::NotThreadLocal,
        SharedAddressSpace);
  }

  // Get the CUDA thread id of the current OpenMP thread on the GPU.
  auto *ThreadID = getNVPTXThreadID(CGF);
  // nvptx_lane_id = nvptx_id % warpsize
  auto *LaneID = getNVPTXLaneID(CGF);
  // nvptx_warp_id = nvptx_id / warpsize
  auto *WarpID = getNVPTXWarpID(CGF);

  Address AddrReduceListArg = CGF.GetAddrOfLocalVar(&ReduceListArg);
  Address LocalReduceList(
      Bld.CreatePointerBitCastOrAddrSpaceCast(
          CGF.EmitLoadOfScalar(AddrReduceListArg, /*Volatile=*/false,
                               C.VoidPtrTy, SourceLocation()),
          CGF.ConvertTypeForMem(ReductionArrayTy)->getPointerTo()),
      CGF.getPointerAlign());

  unsigned Idx = 0;
  for (auto &Private : Privates) {
    //
    // Warp master copies reduce element to transfer medium in __shared__
    // memory.
    //
    llvm::BasicBlock *ThenBB = CGF.createBasicBlock("then");
    llvm::BasicBlock *ElseBB = CGF.createBasicBlock("else");
    llvm::BasicBlock *MergeBB = CGF.createBasicBlock("ifcont");

    // if (lane_id == 0)
    auto IsWarpMaster =
        Bld.CreateICmpEQ(LaneID, Bld.getInt32(0), "warp_master");
    Bld.CreateCondBr(IsWarpMaster, ThenBB, ElseBB);
    CGF.EmitBlock(ThenBB);

    // Reduce element = LocalReduceList[i]
    Address ElemPtrPtrAddr =
        Bld.CreateConstArrayGEP(LocalReduceList, Idx, CGF.getPointerSize());
    llvm::Value *ElemPtrPtr = CGF.EmitLoadOfScalar(
        ElemPtrPtrAddr, /*Volatile=*/false, C.VoidPtrTy, SourceLocation());
    // elemptr = (type[i]*)(elemptrptr)
    Address ElemPtr =
        Address(ElemPtrPtr, C.getTypeAlignInChars(Private->getType()));
    ElemPtr = Bld.CreateElementBitCast(
        ElemPtr, CGF.ConvertTypeForMem(Private->getType()));
    // elem = *elemptr
    llvm::Value *Elem = CGF.EmitLoadOfScalar(
        ElemPtr, /*Volatile=*/false, Private->getType(), SourceLocation());

    // Get pointer to location in transfer medium.
    // MediumPtr = &medium[warp_id]
    llvm::Value *MediumPtrVal = Bld.CreateInBoundsGEP(
        TransferMedium, {llvm::Constant::getNullValue(CGM.Int64Ty), WarpID});
    Address MediumPtr(MediumPtrVal, C.getTypeAlignInChars(Private->getType()));
    // Casting to actual data type.
    // MediumPtr = (type[i]*)MediumPtrAddr;
    MediumPtr = Bld.CreateElementBitCast(
        MediumPtr, CGF.ConvertTypeForMem(Private->getType()));

    //*MediumPtr = elem
    Bld.CreateStore(Elem, MediumPtr);

    Bld.CreateBr(MergeBB);

    CGF.EmitBlock(ElseBB);
    Bld.CreateBr(MergeBB);

    CGF.EmitBlock(MergeBB);

    Address AddrNumWarpsArg = CGF.GetAddrOfLocalVar(&NumWarpsArg);
    llvm::Value *NumWarpsVal = CGF.EmitLoadOfScalar(
        AddrNumWarpsArg, /*Volatile=*/false, C.IntTy, SourceLocation());

    auto *NumActiveThreads = Bld.CreateNSWMul(
        NumWarpsVal, getNVPTXWarpSize(CGF), "num_active_threads");
    // named_barrier_sync(ParallelBarrierID, num_active_threads)
    syncParallelThreads(CGF, NumActiveThreads);

    //
    // Warp 0 copies reduce element from transfer medium.
    //
    llvm::BasicBlock *W0ThenBB = CGF.createBasicBlock("then");
    llvm::BasicBlock *W0ElseBB = CGF.createBasicBlock("else");
    llvm::BasicBlock *W0MergeBB = CGF.createBasicBlock("ifcont");

    // Up to 32 threads in warp 0 are active.
    auto IsActiveThread =
        Bld.CreateICmpULT(ThreadID, NumWarpsVal, "is_active_thread");
    Bld.CreateCondBr(IsActiveThread, W0ThenBB, W0ElseBB);

    CGF.EmitBlock(W0ThenBB);

    // SrcMediumPtr = &medium[tid]
    llvm::Value *SrcMediumPtrVal = Bld.CreateInBoundsGEP(
        TransferMedium, {llvm::Constant::getNullValue(CGM.Int64Ty), ThreadID});
    Address SrcMediumPtr(SrcMediumPtrVal,
                         C.getTypeAlignInChars(Private->getType()));
    // SrcMediumVal = *SrcMediumPtr;
    SrcMediumPtr = Bld.CreateElementBitCast(
        SrcMediumPtr, CGF.ConvertTypeForMem(Private->getType()));
    llvm::Value *SrcMediumValue = CGF.EmitLoadOfScalar(
        SrcMediumPtr, /*Volatile=*/false, Private->getType(), SourceLocation());

    // TargetElemPtr = (type[i]*)(SrcDataAddr[i])
    Address TargetElemPtrPtr =
        Bld.CreateConstArrayGEP(LocalReduceList, Idx, CGF.getPointerSize());
    llvm::Value *TargetElemPtrVal = CGF.EmitLoadOfScalar(
        TargetElemPtrPtr, /*Volatile=*/false, C.VoidPtrTy, SourceLocation());
    Address TargetElemPtr =
        Address(TargetElemPtrVal, C.getTypeAlignInChars(Private->getType()));
    TargetElemPtr = Bld.CreateElementBitCast(
        TargetElemPtr, CGF.ConvertTypeForMem(Private->getType()));

    // *TargetElemPtr = SrcMediumVal;
    CGF.EmitStoreOfScalar(SrcMediumValue, TargetElemPtr, /*Volatile=*/false,
                          Private->getType());
    Bld.CreateBr(W0MergeBB);

    CGF.EmitBlock(W0ElseBB);
    Bld.CreateBr(W0MergeBB);

    CGF.EmitBlock(W0MergeBB);

    // While warp 0 copies values from transfer medium, all other warps must
    // wait.
    syncParallelThreads(CGF, NumActiveThreads);
    Idx++;
  }

  CGF.FinishFunction();
  return Fn;
}

/// Emit a helper that reduces data across two OpenMP threads (lanes)
/// in the same warp.  It uses shuffle instructions to copy over data from
/// a remote lane's stack.  The reduction algorithm performed is specified
/// by the fourth parameter.
///
/// Algorithm Versions.
/// Full Warp Reduce (argument value 0):
///   This algorithm assumes that all 32 lanes are active and gathers
///   data from these 32 lanes, producing a single resultant value.
/// Contiguous Partial Warp Reduce (argument value 1):
///   This algorithm assumes that only a *contiguous* subset of lanes
///   are active.  This happens for the last warp in a parallel region
///   when the user specified num_threads is not an integer multiple of
///   32.  This contiguous subset always starts with the zeroth lane.
/// Partial Warp Reduce (argument value 2):
///   This algorithm gathers data from any number of lanes at any position.
/// All reduced values are stored in the lowest possible lane.  The set
/// of problems every algorithm addresses is a super set of those
/// addressable by algorithms with a lower version number.  Overhead
/// increases as algorithm version increases.
///
/// Terminology
/// Reduce element:
///   Reduce element refers to the individual data field with primitive
///   data types to be combined and reduced across threads.
/// Reduce list:
///   Reduce list refers to a collection of local, thread-private
///   reduce elements.
/// Remote Reduce list:
///   Remote Reduce list refers to a collection of remote (relative to
///   the current thread) reduce elements.
///
/// We distinguish between three states of threads that are important to
/// the implementation of this function.
/// Alive threads:
///   Threads in a warp executing the SIMT instruction, as distinguished from
///   threads that are inactive due to divergent control flow.
/// Active threads:
///   The minimal set of threads that has to be alive upon entry to this
///   function.  The computation is correct iff active threads are alive.
///   Some threads are alive but they are not active because they do not
///   contribute to the computation in any useful manner.  Turning them off
///   may introduce control flow overheads without any tangible benefits.
/// Effective threads:
///   In order to comply with the argument requirements of the shuffle
///   function, we must keep all lanes holding data alive.  But at most
///   half of them perform value aggregation; we refer to this half of
///   threads as effective. The other half is simply handing off their
///   data.
///
/// Procedure
/// Value shuffle:
///   In this step active threads transfer data from higher lane positions
///   in the warp to lower lane positions, creating Remote Reduce list.
/// Value aggregation:
///   In this step, effective threads combine their thread local Reduce list
///   with Remote Reduce list and store the result in the thread local
///   Reduce list.
/// Value copy:
///   In this step, we deal with the assumption made by algorithm 2
///   (i.e. contiguity assumption).  When we have an odd number of lanes
///   active, say 2k+1, only k threads will be effective and therefore k
///   new values will be produced.  However, the Reduce list owned by the
///   (2k+1)th thread is ignored in the value aggregation.  Therefore
///   we copy the Reduce list from the (2k+1)th lane to (k+1)th lane so
///   that the contiguity assumption still holds.
static llvm::Value *emitShuffleAndReduceFunction(
    CodeGenModule &CGM, ArrayRef<const Expr *> Privates,
    QualType ReductionArrayTy, llvm::Value *ReduceFn, SourceLocation Loc) {
  auto &C = CGM.getContext();

  // Thread local Reduce list used to host the values of data to be reduced.
  ImplicitParamDecl ReduceListArg(C, /*DC=*/nullptr, Loc, /*Id=*/nullptr,
                                  C.VoidPtrTy, ImplicitParamDecl::Other);
  // Current lane id; could be logical.
  ImplicitParamDecl LaneIDArg(C, /*DC=*/nullptr, Loc, /*Id=*/nullptr, C.ShortTy,
                              ImplicitParamDecl::Other);
  // Offset of the remote source lane relative to the current lane.
  ImplicitParamDecl RemoteLaneOffsetArg(C, /*DC=*/nullptr, Loc, /*Id=*/nullptr,
                                        C.ShortTy, ImplicitParamDecl::Other);
  // Algorithm version.  This is expected to be known at compile time.
  ImplicitParamDecl AlgoVerArg(C, /*DC=*/nullptr, Loc, /*Id=*/nullptr,
                               C.ShortTy, ImplicitParamDecl::Other);
  FunctionArgList Args;
  Args.push_back(&ReduceListArg);
  Args.push_back(&LaneIDArg);
  Args.push_back(&RemoteLaneOffsetArg);
  Args.push_back(&AlgoVerArg);

  auto &CGFI = CGM.getTypes().arrangeBuiltinFunctionDeclaration(C.VoidTy, Args);
  auto *Fn = llvm::Function::Create(
      CGM.getTypes().GetFunctionType(CGFI), llvm::GlobalValue::InternalLinkage,
      "_omp_reduction_shuffle_and_reduce_func", &CGM.getModule());
  CGM.SetInternalFunctionAttributes(/*D=*/nullptr, Fn, CGFI);
  CodeGenFunction CGF(CGM);
  CGF.StartFunction(GlobalDecl(), C.VoidTy, Fn, CGFI, Args, Loc, Loc);

  auto &Bld = CGF.Builder;

  Address AddrReduceListArg = CGF.GetAddrOfLocalVar(&ReduceListArg);
  Address LocalReduceList(
      Bld.CreatePointerBitCastOrAddrSpaceCast(
          CGF.EmitLoadOfScalar(AddrReduceListArg, /*Volatile=*/false,
                               C.VoidPtrTy, SourceLocation()),
          CGF.ConvertTypeForMem(ReductionArrayTy)->getPointerTo()),
      CGF.getPointerAlign());

  Address AddrLaneIDArg = CGF.GetAddrOfLocalVar(&LaneIDArg);
  llvm::Value *LaneIDArgVal = CGF.EmitLoadOfScalar(
      AddrLaneIDArg, /*Volatile=*/false, C.ShortTy, SourceLocation());

  Address AddrRemoteLaneOffsetArg = CGF.GetAddrOfLocalVar(&RemoteLaneOffsetArg);
  llvm::Value *RemoteLaneOffsetArgVal = CGF.EmitLoadOfScalar(
      AddrRemoteLaneOffsetArg, /*Volatile=*/false, C.ShortTy, SourceLocation());

  Address AddrAlgoVerArg = CGF.GetAddrOfLocalVar(&AlgoVerArg);
  llvm::Value *AlgoVerArgVal = CGF.EmitLoadOfScalar(
      AddrAlgoVerArg, /*Volatile=*/false, C.ShortTy, SourceLocation());

  // Create a local thread-private variable to host the Reduce list
  // from a remote lane.
  Address RemoteReduceList =
      CGF.CreateMemTemp(ReductionArrayTy, ".omp.reduction.remote_reduce_list");

  // This loop iterates through the list of reduce elements and copies,
  // element by element, from a remote lane in the warp to RemoteReduceList,
  // hosted on the thread's stack.
  emitReductionListCopy(RemoteLaneToThread, CGF, ReductionArrayTy, Privates,
                        LocalReduceList, RemoteReduceList,
                        {/*RemoteLaneOffset=*/RemoteLaneOffsetArgVal,
                         /*ScratchpadIndex=*/nullptr,
                         /*ScratchpadWidth=*/nullptr});

  // The actions to be performed on the Remote Reduce list is dependent
  // on the algorithm version.
  //
  //  if (AlgoVer==0) || (AlgoVer==1 && (LaneId < Offset)) || (AlgoVer==2 &&
  //  LaneId % 2 == 0 && Offset > 0):
  //    do the reduction value aggregation
  //
  //  The thread local variable Reduce list is mutated in place to host the
  //  reduced data, which is the aggregated value produced from local and
  //  remote lanes.
  //
  //  Note that AlgoVer is expected to be a constant integer known at compile
  //  time.
  //  When AlgoVer==0, the first conjunction evaluates to true, making
  //    the entire predicate true during compile time.
  //  When AlgoVer==1, the second conjunction has only the second part to be
  //    evaluated during runtime.  Other conjunctions evaluates to false
  //    during compile time.
  //  When AlgoVer==2, the third conjunction has only the second part to be
  //    evaluated during runtime.  Other conjunctions evaluates to false
  //    during compile time.
  auto CondAlgo0 = Bld.CreateICmpEQ(AlgoVerArgVal, Bld.getInt16(0));

  auto Algo1 = Bld.CreateICmpEQ(AlgoVerArgVal, Bld.getInt16(1));
  auto CondAlgo1 = Bld.CreateAnd(
      Algo1, Bld.CreateICmpULT(LaneIDArgVal, RemoteLaneOffsetArgVal));

  auto Algo2 = Bld.CreateICmpEQ(AlgoVerArgVal, Bld.getInt16(2));
  auto CondAlgo2 = Bld.CreateAnd(
      Algo2,
      Bld.CreateICmpEQ(Bld.CreateAnd(LaneIDArgVal, Bld.getInt16(1)),
                       Bld.getInt16(0)));
  CondAlgo2 = Bld.CreateAnd(
      CondAlgo2, Bld.CreateICmpSGT(RemoteLaneOffsetArgVal, Bld.getInt16(0)));

  auto CondReduce = Bld.CreateOr(CondAlgo0, CondAlgo1);
  CondReduce = Bld.CreateOr(CondReduce, CondAlgo2);

  llvm::BasicBlock *ThenBB = CGF.createBasicBlock("then");
  llvm::BasicBlock *ElseBB = CGF.createBasicBlock("else");
  llvm::BasicBlock *MergeBB = CGF.createBasicBlock("ifcont");
  Bld.CreateCondBr(CondReduce, ThenBB, ElseBB);

  CGF.EmitBlock(ThenBB);
  // reduce_function(LocalReduceList, RemoteReduceList)
  llvm::Value *LocalReduceListPtr = Bld.CreatePointerBitCastOrAddrSpaceCast(
      LocalReduceList.getPointer(), CGF.VoidPtrTy);
  llvm::Value *RemoteReduceListPtr = Bld.CreatePointerBitCastOrAddrSpaceCast(
      RemoteReduceList.getPointer(), CGF.VoidPtrTy);
  CGM.getOpenMPRuntime().emitOutlinedFunctionCall(
      CGF, Loc, ReduceFn, {LocalReduceListPtr, RemoteReduceListPtr});
  Bld.CreateBr(MergeBB);

  CGF.EmitBlock(ElseBB);
  Bld.CreateBr(MergeBB);

  CGF.EmitBlock(MergeBB);

  // if (AlgoVer==1 && (LaneId >= Offset)) copy Remote Reduce list to local
  // Reduce list.
  Algo1 = Bld.CreateICmpEQ(AlgoVerArgVal, Bld.getInt16(1));
  auto CondCopy = Bld.CreateAnd(
      Algo1, Bld.CreateICmpUGE(LaneIDArgVal, RemoteLaneOffsetArgVal));

  llvm::BasicBlock *CpyThenBB = CGF.createBasicBlock("then");
  llvm::BasicBlock *CpyElseBB = CGF.createBasicBlock("else");
  llvm::BasicBlock *CpyMergeBB = CGF.createBasicBlock("ifcont");
  Bld.CreateCondBr(CondCopy, CpyThenBB, CpyElseBB);

  CGF.EmitBlock(CpyThenBB);
  emitReductionListCopy(ThreadCopy, CGF, ReductionArrayTy, Privates,
                        RemoteReduceList, LocalReduceList);
  Bld.CreateBr(CpyMergeBB);

  CGF.EmitBlock(CpyElseBB);
  Bld.CreateBr(CpyMergeBB);

  CGF.EmitBlock(CpyMergeBB);

  CGF.FinishFunction();
  return Fn;
}

///
/// Design of OpenMP reductions on the GPU
///
/// Consider a typical OpenMP program with one or more reduction
/// clauses:
///
/// float foo;
/// double bar;
/// #pragma omp target teams distribute parallel for \
///             reduction(+:foo) reduction(*:bar)
/// for (int i = 0; i < N; i++) {
///   foo += A[i]; bar *= B[i];
/// }
///
/// where 'foo' and 'bar' are reduced across all OpenMP threads in
/// all teams.  In our OpenMP implementation on the NVPTX device an
/// OpenMP team is mapped to a CUDA threadblock and OpenMP threads
/// within a team are mapped to CUDA threads within a threadblock.
/// Our goal is to efficiently aggregate values across all OpenMP
/// threads such that:
///
///   - the compiler and runtime are logically concise, and
///   - the reduction is performed efficiently in a hierarchical
///     manner as follows: within OpenMP threads in the same warp,
///     across warps in a threadblock, and finally across teams on
///     the NVPTX device.
///
/// Introduction to Decoupling
///
/// We would like to decouple the compiler and the runtime so that the
/// latter is ignorant of the reduction variables (number, data types)
/// and the reduction operators.  This allows a simpler interface
/// and implementation while still attaining good performance.
///
/// Pseudocode for the aforementioned OpenMP program generated by the
/// compiler is as follows:
///
/// 1. Create private copies of reduction variables on each OpenMP
///    thread: 'foo_private', 'bar_private'
/// 2. Each OpenMP thread reduces the chunk of 'A' and 'B' assigned
///    to it and writes the result in 'foo_private' and 'bar_private'
///    respectively.
/// 3. Call the OpenMP runtime on the GPU to reduce within a team
///    and store the result on the team master:
///
///     __kmpc_nvptx_parallel_reduce_nowait(...,
///        reduceData, shuffleReduceFn, interWarpCpyFn)
///
///     where:
///       struct ReduceData {
///         double *foo;
///         double *bar;
///       } reduceData
///       reduceData.foo = &foo_private
///       reduceData.bar = &bar_private
///
///     'shuffleReduceFn' and 'interWarpCpyFn' are pointers to two
///     auxiliary functions generated by the compiler that operate on
///     variables of type 'ReduceData'.  They aid the runtime perform
///     algorithmic steps in a data agnostic manner.
///
///     'shuffleReduceFn' is a pointer to a function that reduces data
///     of type 'ReduceData' across two OpenMP threads (lanes) in the
///     same warp.  It takes the following arguments as input:
///
///     a. variable of type 'ReduceData' on the calling lane,
///     b. its lane_id,
///     c. an offset relative to the current lane_id to generate a
///        remote_lane_id.  The remote lane contains the second
///        variable of type 'ReduceData' that is to be reduced.
///     d. an algorithm version parameter determining which reduction
///        algorithm to use.
///
///     'shuffleReduceFn' retrieves data from the remote lane using
///     efficient GPU shuffle intrinsics and reduces, using the
///     algorithm specified by the 4th parameter, the two operands
///     element-wise.  The result is written to the first operand.
///
///     Different reduction algorithms are implemented in different
///     runtime functions, all calling 'shuffleReduceFn' to perform
///     the essential reduction step.  Therefore, based on the 4th
///     parameter, this function behaves slightly differently to
///     cooperate with the runtime to ensure correctness under
///     different circumstances.
///
///     'InterWarpCpyFn' is a pointer to a function that transfers
///     reduced variables across warps.  It tunnels, through CUDA
///     shared memory, the thread-private data of type 'ReduceData'
///     from lane 0 of each warp to a lane in the first warp.
/// 4. Call the OpenMP runtime on the GPU to reduce across teams.
///    The last team writes the global reduced value to memory.
///
///     ret = __kmpc_nvptx_teams_reduce_nowait(...,
///             reduceData, shuffleReduceFn, interWarpCpyFn,
///             scratchpadCopyFn, loadAndReduceFn)
///
///     'scratchpadCopyFn' is a helper that stores reduced
///     data from the team master to a scratchpad array in
///     global memory.
///
///     'loadAndReduceFn' is a helper that loads data from
///     the scratchpad array and reduces it with the input
///     operand.
///
///     These compiler generated functions hide address
///     calculation and alignment information from the runtime.
/// 5. if ret == 1:
///     The team master of the last team stores the reduced
///     result to the globals in memory.
///     foo += reduceData.foo; bar *= reduceData.bar
///
///
/// Warp Reduction Algorithms
///
/// On the warp level, we have three algorithms implemented in the
/// OpenMP runtime depending on the number of active lanes:
///
/// Full Warp Reduction
///
/// The reduce algorithm within a warp where all lanes are active
/// is implemented in the runtime as follows:
///
/// full_warp_reduce(void *reduce_data,
///                  kmp_ShuffleReductFctPtr ShuffleReduceFn) {
///   for (int offset = WARPSIZE/2; offset > 0; offset /= 2)
///     ShuffleReduceFn(reduce_data, 0, offset, 0);
/// }
///
/// The algorithm completes in log(2, WARPSIZE) steps.
///
/// 'ShuffleReduceFn' is used here with lane_id set to 0 because it is
/// not used therefore we save instructions by not retrieving lane_id
/// from the corresponding special registers.  The 4th parameter, which
/// represents the version of the algorithm being used, is set to 0 to
/// signify full warp reduction.
///
/// In this version, 'ShuffleReduceFn' behaves, per element, as follows:
///
/// #reduce_elem refers to an element in the local lane's data structure
/// #remote_elem is retrieved from a remote lane
/// remote_elem = shuffle_down(reduce_elem, offset, WARPSIZE);
/// reduce_elem = reduce_elem REDUCE_OP remote_elem;
///
/// Contiguous Partial Warp Reduction
///
/// This reduce algorithm is used within a warp where only the first
/// 'n' (n <= WARPSIZE) lanes are active.  It is typically used when the
/// number of OpenMP threads in a parallel region is not a multiple of
/// WARPSIZE.  The algorithm is implemented in the runtime as follows:
///
/// void
/// contiguous_partial_reduce(void *reduce_data,
///                           kmp_ShuffleReductFctPtr ShuffleReduceFn,
///                           int size, int lane_id) {
///   int curr_size;
///   int offset;
///   curr_size = size;
///   mask = curr_size/2;
///   while (offset>0) {
///     ShuffleReduceFn(reduce_data, lane_id, offset, 1);
///     curr_size = (curr_size+1)/2;
///     offset = curr_size/2;
///   }
/// }
///
/// In this version, 'ShuffleReduceFn' behaves, per element, as follows:
///
/// remote_elem = shuffle_down(reduce_elem, offset, WARPSIZE);
/// if (lane_id < offset)
///     reduce_elem = reduce_elem REDUCE_OP remote_elem
/// else
///     reduce_elem = remote_elem
///
/// This algorithm assumes that the data to be reduced are located in a
/// contiguous subset of lanes starting from the first.  When there is
/// an odd number of active lanes, the data in the last lane is not
/// aggregated with any other lane's dat but is instead copied over.
///
/// Dispersed Partial Warp Reduction
///
/// This algorithm is used within a warp when any discontiguous subset of
/// lanes are active.  It is used to implement the reduction operation
/// across lanes in an OpenMP simd region or in a nested parallel region.
///
/// void
/// dispersed_partial_reduce(void *reduce_data,
///                          kmp_ShuffleReductFctPtr ShuffleReduceFn) {
///   int size, remote_id;
///   int logical_lane_id = number_of_active_lanes_before_me() * 2;
///   do {
///       remote_id = next_active_lane_id_right_after_me();
///       # the above function returns 0 of no active lane
///       # is present right after the current lane.
///       size = number_of_active_lanes_in_this_warp();
///       logical_lane_id /= 2;
///       ShuffleReduceFn(reduce_data, logical_lane_id,
///                       remote_id-1-threadIdx.x, 2);
///   } while (logical_lane_id % 2 == 0 && size > 1);
/// }
///
/// There is no assumption made about the initial state of the reduction.
/// Any number of lanes (>=1) could be active at any position.  The reduction
/// result is returned in the first active lane.
///
/// In this version, 'ShuffleReduceFn' behaves, per element, as follows:
///
/// remote_elem = shuffle_down(reduce_elem, offset, WARPSIZE);
/// if (lane_id % 2 == 0 && offset > 0)
///     reduce_elem = reduce_elem REDUCE_OP remote_elem
/// else
///     reduce_elem = remote_elem
///
///
/// Intra-Team Reduction
///
/// This function, as implemented in the runtime call
/// '__kmpc_nvptx_parallel_reduce_nowait', aggregates data across OpenMP
/// threads in a team.  It first reduces within a warp using the
/// aforementioned algorithms.  We then proceed to gather all such
/// reduced values at the first warp.
///
/// The runtime makes use of the function 'InterWarpCpyFn', which copies
/// data from each of the "warp master" (zeroth lane of each warp, where
/// warp-reduced data is held) to the zeroth warp.  This step reduces (in
/// a mathematical sense) the problem of reduction across warp masters in
/// a block to the problem of warp reduction.
///
///
/// Inter-Team Reduction
///
/// Once a team has reduced its data to a single value, it is stored in
/// a global scratchpad array.  Since each team has a distinct slot, this
/// can be done without locking.
///
/// The last team to write to the scratchpad array proceeds to reduce the
/// scratchpad array.  One or more workers in the last team use the helper
/// 'loadAndReduceDataFn' to load and reduce values from the array, i.e.,
/// the k'th worker reduces every k'th element.
///
/// Finally, a call is made to '__kmpc_nvptx_parallel_reduce_nowait' to
/// reduce across workers and compute a globally reduced value.
///
void CGOpenMPRuntimeNVPTX::emitReduction(
    CodeGenFunction &CGF, SourceLocation Loc, ArrayRef<const Expr *> Privates,
    ArrayRef<const Expr *> LHSExprs, ArrayRef<const Expr *> RHSExprs,
    ArrayRef<const Expr *> ReductionOps, ReductionOptionsTy Options) {
  if (!CGF.HaveInsertPoint())
    return;

  bool ParallelReduction = isOpenMPParallelDirective(Options.ReductionKind);
  bool TeamsReduction = isOpenMPTeamsDirective(Options.ReductionKind);
  // FIXME: Add support for simd reduction.
  assert((TeamsReduction || ParallelReduction) &&
         "Invalid reduction selection in emitReduction.");

  auto &C = CGM.getContext();

  // 1. Build a list of reduction variables.
  // void *RedList[<n>] = {<ReductionVars>[0], ..., <ReductionVars>[<n>-1]};
  auto Size = RHSExprs.size();
  for (auto *E : Privates) {
    if (E->getType()->isVariablyModifiedType())
      // Reserve place for array size.
      ++Size;
  }
  llvm::APInt ArraySize(/*unsigned int numBits=*/32, Size);
  QualType ReductionArrayTy =
      C.getConstantArrayType(C.VoidPtrTy, ArraySize, ArrayType::Normal,
                             /*IndexTypeQuals=*/0);
  Address ReductionList =
      CGF.CreateMemTemp(ReductionArrayTy, ".omp.reduction.red_list");
  auto IPriv = Privates.begin();
  unsigned Idx = 0;
  for (unsigned I = 0, E = RHSExprs.size(); I < E; ++I, ++IPriv, ++Idx) {
    Address Elem = CGF.Builder.CreateConstArrayGEP(ReductionList, Idx,
                                                   CGF.getPointerSize());
    CGF.Builder.CreateStore(
        CGF.Builder.CreatePointerBitCastOrAddrSpaceCast(
            CGF.EmitLValue(RHSExprs[I]).getPointer(), CGF.VoidPtrTy),
        Elem);
    if ((*IPriv)->getType()->isVariablyModifiedType()) {
      // Store array size.
      ++Idx;
      Elem = CGF.Builder.CreateConstArrayGEP(ReductionList, Idx,
                                             CGF.getPointerSize());
      llvm::Value *Size = CGF.Builder.CreateIntCast(
          CGF.getVLASize(
                 CGF.getContext().getAsVariableArrayType((*IPriv)->getType()))
              .first,
          CGF.SizeTy, /*isSigned=*/false);
      CGF.Builder.CreateStore(CGF.Builder.CreateIntToPtr(Size, CGF.VoidPtrTy),
                              Elem);
    }
  }

  // 2. Emit reduce_func().
  auto *ReductionFn = emitReductionFunction(
      CGM, Loc, CGF.ConvertTypeForMem(ReductionArrayTy)->getPointerTo(),
      Privates, LHSExprs, RHSExprs, ReductionOps);

  // 4. Build res = __kmpc_reduce{_nowait}(<gtid>, <n>, sizeof(RedList),
  // RedList, shuffle_reduce_func, interwarp_copy_func);
  auto *ThreadId = getThreadID(CGF, Loc);
  auto *ReductionArrayTySize = CGF.getTypeSize(ReductionArrayTy);
  auto *RL = CGF.Builder.CreatePointerBitCastOrAddrSpaceCast(
      ReductionList.getPointer(), CGF.VoidPtrTy);

  auto *ShuffleAndReduceFn = emitShuffleAndReduceFunction(
      CGM, Privates, ReductionArrayTy, ReductionFn, Loc);
  auto *InterWarpCopyFn =
      emitInterWarpCopyFunction(CGM, Privates, ReductionArrayTy, Loc);

  llvm::Value *Res = nullptr;
  if (ParallelReduction) {
    llvm::Value *Args[] = {ThreadId,
                           CGF.Builder.getInt32(RHSExprs.size()),
                           ReductionArrayTySize,
                           RL,
                           ShuffleAndReduceFn,
                           InterWarpCopyFn};

    Res = CGF.EmitRuntimeCall(
        createNVPTXRuntimeFunction(OMPRTL_NVPTX__kmpc_parallel_reduce_nowait),
        Args);
  }

  if (TeamsReduction) {
    auto *ScratchPadCopyFn =
        emitCopyToScratchpad(CGM, Privates, ReductionArrayTy, Loc);
    auto *LoadAndReduceFn = emitReduceScratchpadFunction(
        CGM, Privates, ReductionArrayTy, ReductionFn, Loc);

    llvm::Value *Args[] = {ThreadId,
                           CGF.Builder.getInt32(RHSExprs.size()),
                           ReductionArrayTySize,
                           RL,
                           ShuffleAndReduceFn,
                           InterWarpCopyFn,
                           ScratchPadCopyFn,
                           LoadAndReduceFn};
    Res = CGF.EmitRuntimeCall(
        createNVPTXRuntimeFunction(OMPRTL_NVPTX__kmpc_teams_reduce_nowait),
        Args);
  }

  // 5. Build switch(res)
  auto *DefaultBB = CGF.createBasicBlock(".omp.reduction.default");
  auto *SwInst = CGF.Builder.CreateSwitch(Res, DefaultBB, /*NumCases=*/1);

  // 6. Build case 1: where we have reduced values in the master
  //    thread in each team.
  //    __kmpc_end_reduce{_nowait}(<gtid>);
  //    break;
  auto *Case1BB = CGF.createBasicBlock(".omp.reduction.case1");
  SwInst->addCase(CGF.Builder.getInt32(1), Case1BB);
  CGF.EmitBlock(Case1BB);

  // Add emission of __kmpc_end_reduce{_nowait}(<gtid>);
  llvm::Value *EndArgs[] = {ThreadId};
  auto &&CodeGen = [&Privates, &LHSExprs, &RHSExprs, &ReductionOps,
                    this](CodeGenFunction &CGF, PrePostActionTy &Action) {
    auto IPriv = Privates.begin();
    auto ILHS = LHSExprs.begin();
    auto IRHS = RHSExprs.begin();
    for (auto *E : ReductionOps) {
      emitSingleReductionCombiner(CGF, E, *IPriv, cast<DeclRefExpr>(*ILHS),
                                  cast<DeclRefExpr>(*IRHS));
      ++IPriv;
      ++ILHS;
      ++IRHS;
    }
  };
  RegionCodeGenTy RCG(CodeGen);
  NVPTXActionTy Action(
      nullptr, llvm::None,
      createNVPTXRuntimeFunction(OMPRTL_NVPTX__kmpc_end_reduce_nowait),
      EndArgs);
  RCG.setAction(Action);
  RCG(CGF);
  CGF.EmitBranch(DefaultBB);
  CGF.EmitBlock(DefaultBB, /*IsFinished=*/true);
}

const VarDecl *
CGOpenMPRuntimeNVPTX::translateParameter(const FieldDecl *FD,
                                         const VarDecl *NativeParam) const {
  if (!NativeParam->getType()->isReferenceType())
    return NativeParam;
  QualType ArgType = NativeParam->getType();
  QualifierCollector QC;
  const Type *NonQualTy = QC.strip(ArgType);
  QualType PointeeTy = cast<ReferenceType>(NonQualTy)->getPointeeType();
  if (const auto *Attr = FD->getAttr<OMPCaptureKindAttr>()) {
    if (Attr->getCaptureKind() == OMPC_map) {
      PointeeTy = CGM.getContext().getAddrSpaceQualType(PointeeTy,
                                                        LangAS::opencl_global);
    }
  }
  ArgType = CGM.getContext().getPointerType(PointeeTy);
  QC.addRestrict();
  enum { NVPTX_local_addr = 5 };
  QC.addAddressSpace(getLangASFromTargetAS(NVPTX_local_addr));
  ArgType = QC.apply(CGM.getContext(), ArgType);
  if (isa<ImplicitParamDecl>(NativeParam)) {
    return ImplicitParamDecl::Create(
        CGM.getContext(), /*DC=*/nullptr, NativeParam->getLocation(),
        NativeParam->getIdentifier(), ArgType, ImplicitParamDecl::Other);
  }
  return ParmVarDecl::Create(
      CGM.getContext(),
      const_cast<DeclContext *>(NativeParam->getDeclContext()),
      NativeParam->getLocStart(), NativeParam->getLocation(),
      NativeParam->getIdentifier(), ArgType,
      /*TInfo=*/nullptr, SC_None, /*DefArg=*/nullptr);
}

Address
CGOpenMPRuntimeNVPTX::getParameterAddress(CodeGenFunction &CGF,
                                          const VarDecl *NativeParam,
                                          const VarDecl *TargetParam) const {
  assert(NativeParam != TargetParam &&
         NativeParam->getType()->isReferenceType() &&
         "Native arg must not be the same as target arg.");
  Address LocalAddr = CGF.GetAddrOfLocalVar(TargetParam);
  QualType NativeParamType = NativeParam->getType();
  QualifierCollector QC;
  const Type *NonQualTy = QC.strip(NativeParamType);
  QualType NativePointeeTy = cast<ReferenceType>(NonQualTy)->getPointeeType();
  unsigned NativePointeeAddrSpace =
      CGF.getContext().getTargetAddressSpace(NativePointeeTy);
  QualType TargetTy = TargetParam->getType();
  llvm::Value *TargetAddr = CGF.EmitLoadOfScalar(
      LocalAddr, /*Volatile=*/false, TargetTy, SourceLocation());
  // First cast to generic.
  TargetAddr = CGF.Builder.CreatePointerBitCastOrAddrSpaceCast(
      TargetAddr, TargetAddr->getType()->getPointerElementType()->getPointerTo(
                      /*AddrSpace=*/0));
  // Cast from generic to native address space.
  TargetAddr = CGF.Builder.CreatePointerBitCastOrAddrSpaceCast(
      TargetAddr, TargetAddr->getType()->getPointerElementType()->getPointerTo(
                      NativePointeeAddrSpace));
  Address NativeParamAddr = CGF.CreateMemTemp(NativeParamType);
  CGF.EmitStoreOfScalar(TargetAddr, NativeParamAddr, /*Volatile=*/false,
                        NativeParamType);
  return NativeParamAddr;
}

void CGOpenMPRuntimeNVPTX::emitOutlinedFunctionCall(
    CodeGenFunction &CGF, SourceLocation Loc, llvm::Value *OutlinedFn,
    ArrayRef<llvm::Value *> Args) const {
  SmallVector<llvm::Value *, 4> TargetArgs;
  TargetArgs.reserve(Args.size());
  auto *FnType =
      cast<llvm::FunctionType>(OutlinedFn->getType()->getPointerElementType());
  for (unsigned I = 0, E = Args.size(); I < E; ++I) {
    if (FnType->isVarArg() && FnType->getNumParams() <= I) {
      TargetArgs.append(std::next(Args.begin(), I), Args.end());
      break;
    }
    llvm::Type *TargetType = FnType->getParamType(I);
    llvm::Value *NativeArg = Args[I];
    if (!TargetType->isPointerTy()) {
      TargetArgs.emplace_back(NativeArg);
      continue;
    }
    llvm::Value *TargetArg = CGF.Builder.CreatePointerBitCastOrAddrSpaceCast(
        NativeArg, NativeArg->getType()->getPointerElementType()->getPointerTo(
                       /*AddrSpace=*/0));
    TargetArgs.emplace_back(
        CGF.Builder.CreatePointerBitCastOrAddrSpaceCast(TargetArg, TargetType));
  }
  CGOpenMPRuntime::emitOutlinedFunctionCall(CGF, Loc, OutlinedFn, TargetArgs);
}

/// Emit function which wraps the outline parallel region
/// and controls the arguments which are passed to this function.
/// The wrapper ensures that the outlined function is called
/// with the correct arguments when data is shared.
llvm::Function *CGOpenMPRuntimeNVPTX::createDataSharingWrapper(
    llvm::Function *OutlinedParallelFn, const OMPExecutableDirective &D) {
  ASTContext &Ctx = CGM.getContext();
  const CapturedStmt &CS = *D.getCapturedStmt(OMPD_parallel);

  // Create a function that takes as argument the source thread.
  FunctionArgList WrapperArgs;
  QualType Int16QTy =
      Ctx.getIntTypeForBitwidth(/*DestWidth=*/16, /*Signed=*/false);
  QualType Int32QTy =
      Ctx.getIntTypeForBitwidth(/*DestWidth=*/32, /*Signed=*/false);
  QualType Int32PtrQTy = Ctx.getPointerType(Int32QTy);
  QualType VoidPtrPtrQTy = Ctx.getPointerType(Ctx.VoidPtrTy);
  ImplicitParamDecl ParallelLevelArg(Ctx, /*DC=*/nullptr, D.getLocStart(),
                                     /*Id=*/nullptr, Int16QTy,
                                     ImplicitParamDecl::Other);
  ImplicitParamDecl WrapperArg(Ctx, /*DC=*/nullptr, D.getLocStart(),
                               /*Id=*/nullptr, Int32QTy,
                               ImplicitParamDecl::Other);
  ImplicitParamDecl SharedArgsList(Ctx, /*DC=*/nullptr, D.getLocStart(),
                                   /*Id=*/nullptr, VoidPtrPtrQTy,
                                   ImplicitParamDecl::Other);
  WrapperArgs.emplace_back(&ParallelLevelArg);
  WrapperArgs.emplace_back(&WrapperArg);
  WrapperArgs.emplace_back(&SharedArgsList);

  auto &CGFI =
      CGM.getTypes().arrangeBuiltinFunctionDeclaration(Ctx.VoidTy, WrapperArgs);

  auto *Fn = llvm::Function::Create(
      CGM.getTypes().GetFunctionType(CGFI), llvm::GlobalValue::InternalLinkage,
      OutlinedParallelFn->getName() + "_wrapper", &CGM.getModule());
  CGM.SetInternalFunctionAttributes(/*D=*/nullptr, Fn, CGFI);
  Fn->setLinkage(llvm::GlobalValue::InternalLinkage);

  CodeGenFunction CGF(CGM, /*suppressNewContext=*/true);
  CGF.StartFunction(GlobalDecl(), Ctx.VoidTy, Fn, CGFI, WrapperArgs,
                    D.getLocStart(), D.getLocStart());

  const auto *RD = CS.getCapturedRecordDecl();
  auto CurField = RD->field_begin();

  // Get the array of arguments.
  SmallVector<llvm::Value *, 8> Args;

  // TODO: suppport SIMD and pass actual values
  Args.emplace_back(llvm::ConstantPointerNull::get(
      CGM.Int32Ty->getPointerTo()));
  Args.emplace_back(llvm::ConstantPointerNull::get(
      CGM.Int32Ty->getPointerTo()));

  CGBuilderTy &Bld = CGF.Builder;
  auto CI = CS.capture_begin();

  // Load the start of the array
  auto SharedArgs =
      CGF.EmitLoadOfPointer(CGF.GetAddrOfLocalVar(&SharedArgsList),
          VoidPtrPtrQTy->castAs<PointerType>());

  // For each captured variable
  for (unsigned I = 0; I < CS.capture_size(); ++I, ++CI, ++CurField) {
    // Name of captured variable
    StringRef Name;
    if (CI->capturesThis())
      Name = "this";
    else
      Name = CI->getCapturedVar()->getName();

    // We retrieve the CLANG type of the argument. We use it to create
    // an alloca which will give us the LLVM type.
    QualType ElemTy = CurField->getType();
    // If this is a capture by copy the element type has to be the pointer to
    // the data.
    if (CI->capturesVariableByCopy())
      ElemTy = Ctx.getPointerType(ElemTy);

    // Get shared address of the captured variable.
    Address ArgAddress = Bld.CreateConstInBoundsGEP(
        SharedArgs, I, CGF.getPointerSize());
    Address TypedArgAddress = Bld.CreateBitCast(
        ArgAddress, CGF.ConvertTypeForMem(Ctx.getPointerType(ElemTy)));
    llvm::Value *Arg = CGF.EmitLoadOfScalar(TypedArgAddress,
        /*Volatile=*/false, Int32PtrQTy, SourceLocation());
    Args.emplace_back(Arg);
  }

  emitOutlinedFunctionCall(CGF, D.getLocStart(), OutlinedParallelFn, Args);
  CGF.FinishFunction();
  return Fn;
}
