//===--- CGCilk.cpp - Emit LLVM Code for Cilk expressions -----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This contains code dealing with code generation of Cilk statements and
// expressions.
//
//===----------------------------------------------------------------------===//

#include "CodeGenFunction.h"
#include "CGCleanup.h"
#include "clang/AST/ExprCilk.h"
#include "clang/AST/StmtCilk.h"

using namespace clang;
using namespace CodeGen;

llvm::Instruction *CodeGenFunction::EmitSyncRegionStart() {
  // Start the sync region.  To ensure the syncregion.start call dominates all
  // uses of the generated token, we insert this call at the alloca insertion
  // point.
  llvm::Instruction *SRStart = llvm::CallInst::Create(
      CGM.getIntrinsic(llvm::Intrinsic::syncregion_start),
      "syncreg", AllocaInsertPt);
  return SRStart;
}

/// EmitCilkSyncStmt - Emit a _Cilk_sync node.
void CodeGenFunction::EmitCilkSyncStmt(const CilkSyncStmt &S) {
  llvm::BasicBlock *ContinueBlock = createBasicBlock("sync.continue");

  // If this code is reachable then emit a stop point (if generating
  // debug info). We have to do this ourselves because we are on the
  // "simple" statement path.
  if (HaveInsertPoint())
    EmitStopPoint(&S);

  EnsureSyncRegion();

  llvm::Instruction *SRStart = CurSyncRegion->getSyncRegionStart();

  Builder.CreateSync(ContinueBlock, SRStart);
  EmitBlock(ContinueBlock);
}

/// \brief Cleanup to ensure parent stack frame is synced.
struct ImplicitSyncCleanup : public EHScopeStack::Cleanup {
public:
  ImplicitSyncCleanup() {}
  void Emit(CodeGenFunction &CGF, Flags F) {
    if (F.isForEHCleanup()) {
      llvm::BasicBlock *ContinueBlock = CGF.createBasicBlock("sync.continue");
      CGF.Builder.CreateSync(ContinueBlock,
                             CGF.CurSyncRegion->getSyncRegionStart());
      CGF.EmitBlock(ContinueBlock);
    }
  }
};

/// \brief Cleanup to ensure parent stack frame is synced.
struct RethrowCleanup : public EHScopeStack::Cleanup {
  llvm::BasicBlock *InvokeDest;
public:
  RethrowCleanup(llvm::BasicBlock *InvokeDest = nullptr)
      : InvokeDest(InvokeDest) {}
  void Emit(CodeGenFunction &CGF, Flags F) {
    llvm::BasicBlock *DetRethrowBlock = CGF.createBasicBlock("det.rethrow");
    if (InvokeDest)
      CGF.Builder.CreateInvoke(
          CGF.CGM.getIntrinsic(llvm::Intrinsic::detached_rethrow),
          DetRethrowBlock, InvokeDest);
    else
      CGF.Builder.CreateBr(DetRethrowBlock);
    CGF.EmitBlock(DetRethrowBlock);
  }
};

// TODO: When a _Cilk_spawn or _Cilk_for appears withiin a try-catch block and
// the spawned computation can throw, add an implicit sync cleanup for the
// spawned computation.  This cleanup path should appear as the unwind
// destination of the generated detach.

bool CodeGenFunction::GenerateStartOfCilkSpawn() {
  // Check if the exception from this detach might be caught.
  // TODO: Replace this code with something cleaner?
  EHScopeStack::iterator I = EHStack.begin(), E = EHStack.end();
  for ( ; I != E; ++I)
    if (EHScope::Catch == I->getKind() ||
        EHScope::Filter == I->getKind())
      break;
  bool NonTrivialEH = (I != E);

  if (NonTrivialEH)
    llvm::dbgs() << "Non-trivial exception handling for cilk_spawn.\n";

  if (NonTrivialEH) {
    // The spawned function throws an exception that gets caught.  We need to
    // handle the EH stack in a special manner.
    //
    // TODO: Replace this catch-all with a special cleanup block that can be
    // separated from ordinary EH block set.
    EHCatchScope *CatchScope = EHStack.pushCatch(1);
    CatchScope->setCatchAllHandler(0, getEHResumeBlock(false));
  }

  // EHStack.pushCleanup<ImplicitSyncCleanup>(EHCleanup);
  return NonTrivialEH;
}

void CodeGenFunction::EmitCilkSpawnStmt(const CilkSpawnStmt &S) {
  // Handle spawning of calls in a special manner, to evaluate
  // arguments before spawn.
  if (const CallExpr *CE = dyn_cast<CallExpr>(S.getSpawnedStmt())) {
    // Set up to perform a detach.
    assert(!IsSpawned &&
           "_Cilk_spawn statement found in spawning environment.");
    IsSpawned = true;
    PushDetachScope();

    // Emit the call.
    EmitCallExpr(CE);

    // Finish the detach.
    assert(CurDetachScope->IsDetachStarted() &&
           "Processing _Cilk_spawn of expression did not produce a detach.");
    IsSpawned = false;
    PopDetachScope();

    return;
  }

  // Otherwise, we assume that the programmer dealt with races correctly.

  // Set up to perform a detach.
  PushDetachScope();
  CurDetachScope->StartDetach();

  // Emit the spawned statement.
  EmitStmt(S.getSpawnedStmt());

  // Finish the detach.
  PopDetachScope();
}

// Helper routine copied from CodeGenFunction.cpp
static void EmitIfUsed(CodeGenFunction &CGF, llvm::BasicBlock *BB) {
  if (!BB) return;
  if (!BB->use_empty())
    return CGF.CurFn->getBasicBlockList().push_back(BB);
  delete BB;
}

void CodeGenFunction::EmitCilkForStmt(const CilkForStmt &S,
                                      ArrayRef<const Attr *> ForAttrs) {
  JumpDest LoopExit = getJumpDestInCurrentScope("pfor.end");

  PushSyncRegion();
  llvm::Instruction *SyncRegionStart = EmitSyncRegionStart();
  CurSyncRegion->setSyncRegionStart(SyncRegionStart);

  LexicalScope ForScope(*this, S.getSourceRange());

  // Evaluate the first part before the loop.
  if (S.getInit())
    EmitStmt(S.getInit());

  assert(S.getCond() && "_Cilk_for loop has no condition");

  // Start the loop with a block that tests the condition.
  // If there's an increment, the continue scope will be overwritten
  // later.
  JumpDest Continue = getJumpDestInCurrentScope("pfor.cond");
  llvm::BasicBlock *CondBlock = Continue.getBlock();
  EmitBlock(CondBlock);

  LoopStack.setSpawnStrategy(LoopAttributes::DAC);
  const SourceRange &R = S.getSourceRange();
  LoopStack.push(CondBlock, CGM.getContext(), ForAttrs,
                 SourceLocToDebugLoc(R.getBegin()),
                 SourceLocToDebugLoc(R.getEnd()));

  const Expr *Inc = S.getInc();
  assert(Inc && "_Cilk_for loop has no increment");
  //llvm::BasicBlock *Preattach = createBasicBlock("pfor.preattach");
  //llvm::errs() << (void*) Preattach << "\n";
  JumpDest Preattach = getJumpDestInCurrentScope("pfor.preattach");
  Continue = getJumpDestInCurrentScope("pfor.inc");

  // Store the blocks to use for break and continue.
  BreakContinueStack.push_back(BreakContinue(Preattach, Preattach));

  // Create a cleanup scope for the condition variable cleanups.
  LexicalScope ConditionScope(*this, S.getSourceRange());

  // Save the old alloca insert point.
  llvm::AssertingVH<llvm::Instruction> OldAllocaInsertPt = AllocaInsertPt;
  // Save the old EH state.
  llvm::BasicBlock *OldEHResumeBlock = EHResumeBlock;
  llvm::Value *OldExceptionSlot = ExceptionSlot;
  llvm::AllocaInst *OldEHSelectorSlot = EHSelectorSlot;

  llvm::BasicBlock *SyncContinueBlock = createBasicBlock("pfor.end.continue");
  bool madeSync = false;
  const VarDecl *LoopVar = S.getLoopVariable();
  RValue LoopVarInitRV;
  llvm::BasicBlock *DetachBlock;
  llvm::BasicBlock *ForBodyEntry;
  llvm::BasicBlock *ForBody;
  {
    // // If the for statement has a condition scope, emit the local variable
    // // declaration.
    // if (S.getConditionVariable()) {
    //   EmitAutoVarDecl(*S.getConditionVariable());
    // }

    llvm::BasicBlock *ExitBlock = LoopExit.getBlock();
    // If there are any cleanups between here and the loop-exit scope,
    // create a block to stage a loop exit along.
    if (ForScope.requiresCleanups())
      ExitBlock = createBasicBlock("pfor.cond.cleanup");

    // As long as the condition is true, iterate the loop.
    DetachBlock = createBasicBlock("pfor.detach");
    // Emit extra entry block for detached body, to ensure that this detached
    // entry block has just one predecessor.
    ForBodyEntry = createBasicBlock("pfor.body.entry");
    ForBody = createBasicBlock("pfor.body");

    // C99 6.8.5p2/p4: The first substatement is executed if the expression
    // compares unequal to 0.  The condition must be a scalar type.
    llvm::Value *BoolCondVal = EvaluateExprAsBool(S.getCond());
    Builder.CreateCondBr(
        BoolCondVal, DetachBlock, ExitBlock,
        createProfileWeightsForLoop(S.getCond(), getProfileCount(S.getBody())));

    if (ExitBlock != LoopExit.getBlock()) {
      EmitBlock(ExitBlock);
      Builder.CreateSync(SyncContinueBlock, SyncRegionStart);
      EmitBlock(SyncContinueBlock);
      PopSyncRegion();
      madeSync = true;
      EmitBranchThroughCleanup(LoopExit);
    }

    EmitBlock(DetachBlock);

    // Get the value of the loop variable initialization before we emit the
    // detach.
    if (LoopVar)
      LoopVarInitRV = EmitAnyExprToTemp(LoopVar->getInit());

    Builder.CreateDetach(ForBodyEntry, Continue.getBlock(), SyncRegionStart);

    // Create a new alloca insert point.
    llvm::Value *Undef = llvm::UndefValue::get(Int32Ty);
    AllocaInsertPt = new llvm::BitCastInst(Undef, Int32Ty, "", ForBodyEntry);
    // Set up nested EH state.
    EHResumeBlock = nullptr;
    ExceptionSlot = nullptr;
    EHSelectorSlot = nullptr;

    EmitBlock(ForBodyEntry);
  }

  // Create a cleanup scope for the loop-variable cleanups.
  RunCleanupsScope DetachCleanupsScope(*this);
  EHStack.pushCleanup<RethrowCleanup>(EHCleanup);

  // Inside the detached block, create the loop variable, setting its value to
  // the saved initialization value.
  if (LoopVar) {
    AutoVarEmission LVEmission = EmitAutoVarAlloca(*LoopVar);
    QualType type = LoopVar->getType();
    Address Loc = LVEmission.getObjectAddress(*this);
    LValue LV = MakeAddrLValue(Loc, type);
    LV.setNonGC(true);
    EmitStoreThroughLValue(LoopVarInitRV, LV, true);
    EmitAutoVarCleanups(LVEmission);
  }

  Builder.CreateBr(ForBody);

  EmitBlock(ForBody);

  incrementProfileCounter(&S);

  {
    // Create a separate cleanup scope for the body, in case it is not
    // a compound statement.
    RunCleanupsScope BodyScope(*this);
    EmitStmt(S.getBody());
    Builder.CreateBr(Preattach.getBlock());
  }

  // Finish detached body and emit the reattach.
  {
    EmitBlock(Preattach.getBlock());

    DetachCleanupsScope.ForceCleanup();

    Builder.CreateReattach(Continue.getBlock(), SyncRegionStart);
  }

  // Restore CGF state after detached region.
  {
    // Restore the alloca insertion point.
    llvm::Instruction *Ptr = AllocaInsertPt;
    AllocaInsertPt = OldAllocaInsertPt;
    Ptr->eraseFromParent();

    // Restore the EH state.
    EmitIfUsed(*this, EHResumeBlock);
    EHResumeBlock = OldEHResumeBlock;
    ExceptionSlot = OldExceptionSlot;
    EHSelectorSlot = OldEHSelectorSlot;
  }

  // Emit the increment next.
  EmitBlock(Continue.getBlock());
  EmitStmt(Inc);

  BreakContinueStack.pop_back();

  ConditionScope.ForceCleanup();

  EmitStopPoint(&S);
  EmitBranch(CondBlock);

  ForScope.ForceCleanup();

  LoopStack.pop();
  // Emit the fall-through block.
  EmitBlock(LoopExit.getBlock(), true);
  if (!madeSync) {
    Builder.CreateSync(SyncContinueBlock, SyncRegionStart);
    EmitBlock(SyncContinueBlock);
    PopSyncRegion();
  }
}
