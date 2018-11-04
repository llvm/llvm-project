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

// Stolen from CodeGenFunction.cpp
static void EmitIfUsed(CodeGenFunction &CGF, llvm::BasicBlock *BB) {
  if (!BB) return;
  if (!BB->use_empty())
    return CGF.CurFn->getBasicBlockList().push_back(BB);
  delete BB;
}

CodeGenFunction::IsSpawnedScope::IsSpawnedScope(CodeGenFunction *CGF)
    : CGF(CGF), OldIsSpawned(CGF->IsSpawned) {
  CGF->IsSpawned = false;
}

CodeGenFunction::IsSpawnedScope::~IsSpawnedScope() {
  RestoreOldScope();
}

bool CodeGenFunction::IsSpawnedScope::OldScopeIsSpawned() {
  return OldIsSpawned;
}

void CodeGenFunction::IsSpawnedScope::RestoreOldScope() {
  CGF->IsSpawned = OldIsSpawned;
}

llvm::BasicBlock *CodeGenFunction::DetachedRethrowHandler::get() {
  if (DetachedRethrowBlock)
    return DetachedRethrowBlock;

  DetachedRethrowBlock = CGF.createBasicBlock("det.rethrow");
  return DetachedRethrowBlock;
}

void CodeGenFunction::DetachedRethrowHandler::emitIfUsed(
    llvm::Value *ExnSlot, llvm::Value *SelSlot, llvm::Value *SyncRegion) {
  if (!isUsed())
    return;

  DetachedRethrowResumeBlock = CGF.createBasicBlock("det.rethrow.unreachable");

  CGBuilderTy::InsertPoint SavedIP = CGF.Builder.saveIP();
  CGF.Builder.SetInsertPoint(DetachedRethrowBlock);

  // Recreate the landingpad's return value for the rethrow invoke.  Tapir
  // lowering will replace this rethrow with a resume.
  assert(ExnSlot &&
         "DetachedRethrowHandler used with no Exception slot!");
  llvm::Value *Exn = CGF.Builder.CreateLoad(
      Address(ExnSlot, CGF.getPointerAlign()), "exn");
  assert(SelSlot &&
         "DetachedRethrowHandler used with no Selector slot!");
  llvm::Value *Sel = CGF.Builder.CreateLoad(
      Address(SelSlot, CharUnits::fromQuantity(4)), "sel");

  llvm::Type *LPadType = llvm::StructType::get(Exn->getType(),
                                               Sel->getType());
  llvm::Value *LPadVal = llvm::UndefValue::get(LPadType);
  LPadVal = CGF.Builder.CreateInsertValue(LPadVal, Exn, 0, "lpad.val");
  LPadVal = CGF.Builder.CreateInsertValue(LPadVal, Sel, 1, "lpad.val");

  // Insert an invoke of the detached_rethrow intrinsic.
  llvm::BasicBlock *InvokeDest = CGF.getInvokeDest();
  llvm::Function *DetachedRethrow =
    CGF.CGM.getIntrinsic(llvm::Intrinsic::detached_rethrow,
                         { LPadVal->getType() });
  CGF.Builder.CreateInvoke(
      DetachedRethrow, DetachedRethrowResumeBlock, InvokeDest,
      { SyncRegion, LPadVal });

  // The detached_rethrow intrinsic is just a placeholder, so the ordinary
  // destination should of the invoke should be unreachable.
  CGF.Builder.SetInsertPoint(DetachedRethrowResumeBlock);
  CGF.Builder.CreateUnreachable();

  CGF.EmitBlockAfterUses(DetachedRethrowBlock);
  CGF.EmitBlock(DetachedRethrowResumeBlock);

  CGF.Builder.restoreIP(SavedIP);
}

void CodeGenFunction::DetachScope::InitDetachScope() {
  // Create the detached and continue blocks.
  DetachedBlock = CGF.createBasicBlock("det.achd");
  ContinueBlock = CGF.createBasicBlock("det.cont");

  // Set the detached block as the new alloca insertion point.
  OldAllocaInsertPt = CGF.AllocaInsertPt;
  llvm::Value *Undef = llvm::UndefValue::get(CGF.Int32Ty);
  CGF.AllocaInsertPt = new llvm::BitCastInst(Undef, CGF.Int32Ty, "",
                                             DetachedBlock);

  DetachInitialized = true;
}

void CodeGenFunction::DetachScope::RestoreDetachScope() {
  OldAllocaInsertPt = CGF.AllocaInsertPt;
  CGF.AllocaInsertPt = SavedDetachedAllocaInsertPt;
}

void CodeGenFunction::DetachScope::StartDetach() {
  if (!DetachInitialized)
    InitDetachScope();
  else
    RestoreDetachScope();

  // Create the detach
  CleanupsScope = new RunCleanupsScope(CGF);
  CGF.pushCleanupAfterFullExpr<ImplicitSyncCleanup>(
      EHCleanup, CGF.CurSyncRegion->getSyncRegionStart());
  Detach = CGF.Builder.CreateDetach(DetachedBlock, ContinueBlock,
                                    CGF.CurSyncRegion->getSyncRegionStart());

  // Save the old EH state.
  OldEHResumeBlock = CGF.EHResumeBlock;
  CGF.EHResumeBlock = nullptr;
  OldExceptionSlot = CGF.ExceptionSlot;
  CGF.ExceptionSlot = nullptr;
  OldEHSelectorSlot = CGF.EHSelectorSlot;
  CGF.EHSelectorSlot = nullptr;
  OldNormalCleanupDest = CGF.NormalCleanupDest;
  CGF.NormalCleanupDest = Address::invalid();

  // Emit the detached block.
  CGF.EmitBlock(DetachedBlock);

  // Create an EH scope for catching exceptions from the detached task.
  // Ultimately, the detached task might be outlined into a separate helper
  // function.  Hence, if an exception might propagate from the task to its
  // parent, then it needs to be rethrown from this helper.  The
  // detached-rethrow handler models this pattern of rethrowing the exception
  // before outlining occurs.
  EHCatchScope *CatchScope = CGF.EHStack.pushCatch(1);
  CatchScope->setCatchAllHandler(0, DetRethrow.get());

  ExnCleanupsScope = new RunCleanupsScope(CGF);

  CGF.PushSyncRegion();

  // For Cilk, ensure that the detached task is implicitly synced before it
  // returns.
  CGF.CurSyncRegion->addImplicitSync();

  // Initialize lifetime intrinsics for the reference temporary.
  if (RefTmp.isValid()) {
    switch (RefTmpSD) {
    case SD_Automatic:
    case SD_FullExpression:
      if (auto *Size = CGF.EmitLifetimeStart(
              CGF.CGM.getDataLayout().getTypeAllocSize(RefTmp.getElementType()),
              RefTmp.getPointer())) {
        if (RefTmpSD == SD_Automatic)
          CGF.pushCleanupAfterFullExpr<CallLifetimeEnd>(NormalEHLifetimeMarker,
                                                        RefTmp, Size);
        else
          CGF.pushFullExprCleanup<CallLifetimeEnd>(NormalEHLifetimeMarker,
                                                   RefTmp, Size);
      }
      break;
    default:
      break;
    }
  }

  DetachStarted = true;
}

void CodeGenFunction::DetachScope::FinishDetach() {
  assert(DetachStarted &&
         "Attempted to finish a detach that was not started.");

  CGF.PopSyncRegion();

  ExnCleanupsScope->ForceCleanup();
  CGF.popCatchScope();

  // The CFG path into the spawned statement should terminate with a `reattach'.
  CGF.Builder.CreateReattach(ContinueBlock,
                             CGF.CurSyncRegion->getSyncRegionStart());

  // Restore the alloca insertion point.
  llvm::Instruction *Ptr = CGF.AllocaInsertPt;
  CGF.AllocaInsertPt = OldAllocaInsertPt;
  SavedDetachedAllocaInsertPt = nullptr;
  Ptr->eraseFromParent();

  // Restore the EH state.
  llvm::Value *DetExnSlot = CGF.ExceptionSlot;
  llvm::Value *DetSelSlot = CGF.EHSelectorSlot;

  EmitIfUsed(CGF, CGF.EHResumeBlock);
  CGF.EHResumeBlock = OldEHResumeBlock;
  CGF.ExceptionSlot = OldExceptionSlot;
  CGF.EHSelectorSlot = OldEHSelectorSlot;
  CGF.NormalCleanupDest = OldNormalCleanupDest;

  // Emit the continue block.
  CleanupsScope->ForceCleanup();
  CGF.EmitBlock(ContinueBlock);

  DetRethrow.emitIfUsed(DetExnSlot, DetSelSlot,
                        CGF.CurSyncRegion->getSyncRegionStart());
  // If the detached-rethrow handler is used, add an unwind destination to the
  // detach.
  if (DetRethrow.isUsed()) {
    CGBuilderTy::InsertPoint SavedIP = CGF.Builder.saveIP();
    CGF.Builder.SetInsertPoint(Detach);
    // Create the new detach instruction.
    llvm::DetachInst *NewDetach = CGF.Builder.CreateDetach(
        Detach->getDetached(), Detach->getContinue(), CGF.getInvokeDest(),
        Detach->getSyncRegion());
    // Remove the old detach.
    Detach->eraseFromParent();
    Detach = NewDetach;
    CGF.Builder.restoreIP(SavedIP);
  }
}

Address CodeGenFunction::DetachScope::CreateDetachedMemTemp(
    QualType Ty, StorageDuration SD, const Twine &Name) {
  if (!DetachInitialized)
    InitDetachScope();
  else
    RestoreDetachScope();

  // There shouldn't be multiple reference temporaries needed.
  assert(!RefTmp.isValid() &&
         "Already created a reference temporary in this detach scope.");

  // Create the reference temporary
  RefTmp = CGF.CreateMemTemp(Ty, Name);
  RefTmpSD = SD;

  // Save the detached scope
  SavedDetachedAllocaInsertPt = CGF.AllocaInsertPt;
  CGF.AllocaInsertPt = OldAllocaInsertPt;

  return RefTmp;
}

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

  // Start the loop with a block that tests the condition.  If there's an
  // increment, the continue scope will be overwritten later.
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
  Continue = getJumpDestInCurrentScope("pfor.inc");

  // Ensure that the _Cilk_for loop iterations are synced on exit from the loop,
  // whether normally or by an exception.
  EHStack.pushCleanup<ImplicitSyncCleanup>(NormalAndEHCleanup,
                                           SyncRegionStart);

  // Create a cleanup scope for the condition variable cleanups.
  LexicalScope ConditionScope(*this, S.getSourceRange());

  // Save the old alloca insert point.
  llvm::AssertingVH<llvm::Instruction> OldAllocaInsertPt = AllocaInsertPt;
  // Save the old EH state.
  llvm::BasicBlock *OldEHResumeBlock = EHResumeBlock;
  llvm::Value *OldExceptionSlot = ExceptionSlot;
  llvm::AllocaInst *OldEHSelectorSlot = EHSelectorSlot;
  Address OldNormalCleanupDest = NormalCleanupDest;

  const VarDecl *LoopVar = S.getLoopVariable();
  RValue LoopVarInitRV;
  llvm::BasicBlock *DetachBlock;
  llvm::BasicBlock *ForBodyEntry;
  llvm::BasicBlock *ForBody;
  llvm::DetachInst *Detach;
  {
    // FIXME: Figure out if there is a way to support condition variables in
    // Cilk.
    //
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
      EmitBranchThroughCleanup(LoopExit);
    }

    EmitBlock(DetachBlock);

    // Get the value of the loop variable initialization before we emit the
    // detach.
    if (LoopVar)
      LoopVarInitRV = EmitAnyExprToTemp(LoopVar->getInit());

    Detach = Builder.CreateDetach(ForBodyEntry, Continue.getBlock(),
                                  SyncRegionStart);

    // Create a new alloca insert point.
    llvm::Value *Undef = llvm::UndefValue::get(Int32Ty);
    AllocaInsertPt = new llvm::BitCastInst(Undef, Int32Ty, "", ForBodyEntry);
    // Set up nested EH state.
    EHResumeBlock = nullptr;
    ExceptionSlot = nullptr;
    EHSelectorSlot = nullptr;
    NormalCleanupDest = Address::invalid();

    EmitBlock(ForBodyEntry);
  }

  // Create an EH scope for the loop-variable cleanups and exceptions.
  // Ultimately, the loop body might be outlined into a separate helper
  // function.  Hence, if an exception from the loop body might propagate out of
  // the loop, then it must be rethrown from the outlined helper.  The
  // detached-rethrow handler models this pattern of rethrowing the exception
  // before outlining occurs.
  DetachedRethrowHandler DetRethrow(*this);
  EHCatchScope *CatchScope = EHStack.pushCatch(1);
  CatchScope->setCatchAllHandler(0, DetRethrow.get());
  RunCleanupsScope DetachCleanupsScope(*this);

  // Store the blocks to use for break and continue.
  JumpDest Preattach = getJumpDestInCurrentScope("pfor.preattach");
  BreakContinueStack.push_back(BreakContinue(Preattach, Preattach));

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
    popCatchScope();
    Builder.CreateReattach(Continue.getBlock(), SyncRegionStart);
  }

  // Restore CGF state after detached region.
  llvm::Value *DetExnSlot = ExceptionSlot;
  llvm::Value *DetSelSlot = EHSelectorSlot;
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
    NormalCleanupDest = OldNormalCleanupDest;
  }

  // Emit the increment next.
  EmitBlock(Continue.getBlock());
  EmitStmt(Inc);

  {
    DetRethrow.emitIfUsed(DetExnSlot, DetSelSlot, SyncRegionStart);
    // If the detached-rethrow handler is used, add an unwind destination to the
    // detach.
    if (DetRethrow.isUsed()) {
      CGBuilderTy::InsertPoint SavedIP = Builder.saveIP();
      Builder.SetInsertPoint(DetachBlock);
      // Create the new detach instruction.
      llvm::DetachInst *NewDetach = Builder.CreateDetach(
          ForBodyEntry, Continue.getBlock(), getInvokeDest(),
          SyncRegionStart);
      // Remove the old detach.
      Detach->eraseFromParent();
      Detach = NewDetach;
      Builder.restoreIP(SavedIP);
    }
  }

  BreakContinueStack.pop_back();

  ConditionScope.ForceCleanup();

  EmitStopPoint(&S);
  EmitBranch(CondBlock);

  ForScope.ForceCleanup();

  LoopStack.pop();
  // Emit the fall-through block.
  EmitBlock(LoopExit.getBlock(), true);
  PopSyncRegion();
}
