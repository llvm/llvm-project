//===----- CGOpenMPRuntimeNVPTX.h - Interface to OpenMP NVPTX Runtimes ----===//
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

#ifndef LLVM_CLANG_LIB_CODEGEN_CGOPENMPRUNTIMENVPTX_H
#define LLVM_CLANG_LIB_CODEGEN_CGOPENMPRUNTIMENVPTX_H

#include "CGOpenMPRuntime.h"
#include "CodeGenFunction.h"
#include "clang/AST/StmtOpenMP.h"
#include "llvm/IR/CallSite.h"

namespace clang {
namespace CodeGen {

class CGOpenMPRuntimeNVPTX : public CGOpenMPRuntime {
public:
  class EntryFunctionState {
  public:
    llvm::BasicBlock *ExitBB;

    EntryFunctionState() : ExitBB(nullptr){};
  };

  class WorkerFunctionState {
  public:
    llvm::Function *WorkerFn;
    const CGFunctionInfo *CGFI;

    WorkerFunctionState(CodeGenModule &CGM);

  private:
    void createWorkerFunction(CodeGenModule &CGM);
  };

  /// \brief Helper for target entry function. Guide the master and worker
  /// threads to their respective locations.
  void emitEntryHeader(CodeGenFunction &CGF, EntryFunctionState &EST,
                       WorkerFunctionState &WST);

  /// \brief Signal termination of OMP execution.
  void emitEntryFooter(CodeGenFunction &CGF, EntryFunctionState &EST);

private:
  //
  // NVPTX calls.
  //

  /// \brief Get the GPU warp size.
  llvm::Value *getNVPTXWarpSize(CodeGenFunction &CGF);

  /// \brief Get the id of the current thread on the GPU.
  llvm::Value *getNVPTXThreadID(CodeGenFunction &CGF);

  // \brief Get the maximum number of threads in a block of the GPU.
  llvm::Value *getNVPTXNumThreads(CodeGenFunction &CGF);

  /// \brief Get barrier to synchronize all threads in a block.
  void getNVPTXCTABarrier(CodeGenFunction &CGF);

  // \brief Synchronize all GPU threads in a block.
  void syncCTAThreads(CodeGenFunction &CGF);

  //
  // OMP calls.
  //

  /// \brief Get the thread id of the OMP master thread.
  /// The master thread id is the first thread (lane) of the last warp in the
  /// GPU block.  Warp size is assumed to be some power of 2.
  /// Thread id is 0 indexed.
  /// E.g: If NumThreads is 33, master id is 32.
  ///      If NumThreads is 64, master id is 32.
  ///      If NumThreads is 1024, master id is 992.
  llvm::Value *getMasterThreadID(CodeGenFunction &CGF);

  //
  // Private state and methods.
  //

  // Master-worker control state.
  // Number of requested OMP threads in parallel region.
  llvm::GlobalVariable *ActiveWorkers;
  // Outlined function for the workers to execute.
  llvm::GlobalVariable *WorkID;

  /// \brief Initialize master-worker control state.
  void initializeEnvironment();

  /// \brief Emit the worker function for the current target region.
  void emitWorkerFunction(WorkerFunctionState &WST);

  /// \brief Helper for worker function. Emit body of worker loop.
  void emitWorkerLoop(CodeGenFunction &CGF, WorkerFunctionState &WST);

  /// \brief Returns specified OpenMP runtime function for the current OpenMP
  /// implementation.  Specialized for the NVPTX device.
  /// \param Function OpenMP runtime function.
  /// \return Specified function.
  llvm::Constant *createNVPTXRuntimeFunction(unsigned Function);

  //
  // Base class overrides.
  //

  /// \brief Creates offloading entry for the provided entry ID \a ID,
  /// address \a Addr and size \a Size.
  void createOffloadEntry(llvm::Constant *ID, llvm::Constant *Addr,
                          uint64_t Size) override;

  /// \brief Emit outlined function for 'target' directive on the NVPTX
  /// device.
  /// \param D Directive to emit.
  /// \param ParentName Name of the function that encloses the target region.
  /// \param OutlinedFn Outlined function value to be defined by this call.
  /// \param OutlinedFnID Outlined function ID value to be defined by this call.
  /// \param IsOffloadEntry True if the outlined function is an offload entry.
  /// An outlined function may not be an entry if, e.g. the if clause always
  /// evaluates to false.
  void emitTargetOutlinedFunction(const OMPExecutableDirective &D,
                                  StringRef ParentName,
                                  llvm::Function *&OutlinedFn,
                                  llvm::Constant *&OutlinedFnID,
                                  bool IsOffloadEntry,
                                  const RegionCodeGenTy &CodeGen) override;

public:
  explicit CGOpenMPRuntimeNVPTX(CodeGenModule &CGM);

  /// \brief This function ought to emit, in the general case, a call to
  // the openmp runtime kmpc_push_num_teams. In NVPTX backend it is not needed
  // as these numbers are obtained through the PTX grid and block configuration.
  /// \param NumTeams An integer expression of teams.
  /// \param ThreadLimit An integer expression of threads.
  void emitNumTeamsClause(CodeGenFunction &CGF, const Expr *NumTeams,
                          const Expr *ThreadLimit, SourceLocation Loc) override;

  /// \brief Emits inlined function for the specified OpenMP parallel
  //  directive but an inlined function for teams.
  /// \a D. This outlined function has type void(*)(kmp_int32 *ThreadID,
  /// kmp_int32 BoundID, struct context_vars*).
  /// \param D OpenMP directive.
  /// \param ThreadIDVar Variable for thread id in the current OpenMP region.
  /// \param InnermostKind Kind of innermost directive (for simple directives it
  /// is a directive itself, for combined - its innermost directive).
  /// \param CodeGen Code generation sequence for the \a D directive.
  llvm::Value *
  emitParallelOrTeamsOutlinedFunction(const OMPExecutableDirective &D,
                                      const VarDecl *ThreadIDVar,
                                      OpenMPDirectiveKind InnermostKind,
                                      const RegionCodeGenTy &CodeGen) override;

  /// \brief Emits code for teams call of the \a OutlinedFn with
  /// variables captured in a record which address is stored in \a
  /// CapturedStruct.
  /// \param OutlinedFn Outlined function to be run by team masters. Type of
  /// this function is void(*)(kmp_int32 *, kmp_int32, struct context_vars*).
  /// \param CapturedVars A pointer to the record with the references to
  /// variables used in \a OutlinedFn function.
  ///
  void emitTeamsCall(CodeGenFunction &CGF, const OMPExecutableDirective &D,
                     SourceLocation Loc, llvm::Value *OutlinedFn,
                     ArrayRef<llvm::Value *> CapturedVars) override;
};

} // CodeGen namespace.
} // clang namespace.

#endif // LLVM_CLANG_LIB_CODEGEN_CGOPENMPRUNTIMENVPTX_H
