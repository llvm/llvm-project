//===--- CGTapir.cpp - Emit LLVM Code for Tapir expressions -----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This contains code dealing with code generation of Tapir statements and
// expressions.
//
//===----------------------------------------------------------------------===//

#include "CodeGenFunction.h"
#include "CGCleanup.h"
#include "clang/AST/StmtTapir.h"

using namespace clang;
using namespace CodeGen;

// Stolen from CodeGenFunction.cpp
static void EmitIfUsed(CodeGenFunction &CGF, llvm::BasicBlock *BB) {
  if (!BB) return;
  if (!BB->use_empty())
    return CGF.CurFn->getBasicBlockList().push_back(BB);
  delete BB;
}

llvm::Instruction *CodeGenFunction::EmitLabeledSyncRegionStart(StringRef SV) {
  // Start the sync region.  To ensure the syncregion.start call dominates all
  // uses of the generated token, we insert this call at the alloca insertion
  // point.
  llvm::Instruction *SRStart = llvm::CallInst::Create(
      CGM.getIntrinsic(llvm::Intrinsic::syncregion_start), SV, AllocaInsertPt);
  return SRStart;
}

/// EmitSyncStmt - Emit a sync node.
void CodeGenFunction::EmitSyncStmt(const SyncStmt &S) {
  llvm::BasicBlock *ContinueBlock = createBasicBlock("sync.continue");

  // If this code is reachable then emit a stop point (if generating
  // debug info). We have to do this ourselves because we are on the
  // "simple" statement path.
  if (HaveInsertPoint())
    EmitStopPoint(&S);

  Builder.CreateSync(ContinueBlock, 
    getOrCreateLabeledSyncRegion(S.getSyncVar())->getSyncRegionStart());
  EmitBlock(ContinueBlock);
}

void CodeGenFunction::EmitSpawnStmt(const SpawnStmt &S) {
  // Set up to perform a detach.
  // PushDetachScope();
  SyncRegion* SR = getOrCreateLabeledSyncRegion(S.getSyncVar());
  //StartLabeledDetach(SR);

  llvm::BasicBlock* DetachedBlock = createBasicBlock("det.achd");
  llvm::BasicBlock* ContinueBlock = createBasicBlock("det.cont");

  auto OldAllocaInsertPt = AllocaInsertPt; 
  llvm::Value *Undef = llvm::UndefValue::get(Int32Ty);
  AllocaInsertPt = new llvm::BitCastInst(Undef, Int32Ty, "",
                                             DetachedBlock);

  Builder.CreateDetach(DetachedBlock, ContinueBlock,
                           SR->getSyncRegionStart());


  EmitBlock(DetachedBlock);
  EmitStmt(S.getSpawnedStmt());

  Builder.CreateReattach(ContinueBlock,
                             SR->getSyncRegionStart());

  llvm::Instruction* ptr = AllocaInsertPt; 
  AllocaInsertPt = OldAllocaInsertPt; 
  ptr->eraseFromParent(); 

  EmitBlock(ContinueBlock);
}

void CodeGenFunction::EmitForallStmt(const ForallStmt &S,
                                  ArrayRef<const Attr *> ForAttrs) {
  JumpDest LoopExit = getJumpDestInCurrentScope("forall.end");

  LexicalScope ForScope(*this, S.getSourceRange());

  // Evaluate the first part before the loop.
  if (S.getInit())
    EmitStmt(S.getInit());

  // Start the loop with a block that tests the condition.
  // If there's an increment, the continue scope will be overwritten
  // later.
  JumpDest Continue = getJumpDestInCurrentScope("forall.cond");
  llvm::BasicBlock *CondBlock = Continue.getBlock();
  EmitBlock(CondBlock);

  const SourceRange &R = S.getSourceRange();
  LoopStack.push(CondBlock, CGM.getContext(), ForAttrs,
                 SourceLocToDebugLoc(R.getBegin()),
                 SourceLocToDebugLoc(R.getEnd()));

  // If the for loop doesn't have an increment we can just use the
  // condition as the continue block.  Otherwise we'll need to create
  // a block for it (in the current scope, i.e. in the scope of the
  // condition), and that we will become our continue block.
  if (S.getInc())
    Continue = getJumpDestInCurrentScope("forall.inc");

  // Store the blocks to use for break and continue.
  BreakContinueStack.push_back(BreakContinue(LoopExit, Continue));

  // Create a cleanup scope for the condition variable cleanups.
  LexicalScope ConditionScope(*this, S.getSourceRange());

  if (S.getCond()) {
    // If the for statement has a condition scope, emit the local variable
    // declaration.
    if (S.getConditionVariable()) {
      EmitDecl(*S.getConditionVariable());
    }

    llvm::BasicBlock *ExitBlock = LoopExit.getBlock();
    // If there are any cleanups between here and the loop-exit scope,
    // create a block to stage a loop exit along.
    if (ForScope.requiresCleanups())
      ExitBlock = createBasicBlock("forall.cond.cleanup");

    // As long as the condition is true, iterate the loop.
    llvm::BasicBlock *ForallBody = createBasicBlock("forall.body");

    // C99 6.8.5p2/p4: The first substatement is executed if the expression
    // compares unequal to 0.  The condition must be a scalar type.
    llvm::Value *BoolCondVal = EvaluateExprAsBool(S.getCond());
    Builder.CreateCondBr(
        BoolCondVal, ForallBody, ExitBlock,
        createProfileWeightsForLoop(S.getCond(), getProfileCount(S.getBody())));

    if (ExitBlock != LoopExit.getBlock()) {
      EmitBlock(ExitBlock);
      EmitBranchThroughCleanup(LoopExit);
    }

    EmitBlock(ForallBody);
  } else {
    // Treat it as a non-zero constant.  Don't even create a new block for the
    // body, just fall into it.
  }
  incrementProfileCounter(&S);

  {
    // Create a separate cleanup scope for the body, in case it is not
    // a compound statement.
    RunCleanupsScope BodyScope(*this);
    EmitStmt(S.getBody());
  }

  // If there is an increment, emit it next.
  if (S.getInc()) {
    EmitBlock(Continue.getBlock());
    EmitStmt(S.getInc());
  }

  BreakContinueStack.pop_back();

  ConditionScope.ForceCleanup();

  EmitStopPoint(&S);
  EmitBranch(CondBlock);

  ForScope.ForceCleanup();

  LoopStack.pop();

  // Emit the fall-through block.
  EmitBlock(LoopExit.getBlock(), true);
}

