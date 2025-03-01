//===--- CIRGenCleanup.cpp - Bookkeeping and code emission for cleanups ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains code dealing with the IR generation for cleanups
// and related information.
//
// A "cleanup" is a piece of code which needs to be executed whenever
// control transfers out of a particular scope.  This can be
// conditionalized to occur only on exceptional control flow, only on
// normal control flow, or both.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/SaveAndRestore.h"

#include "CIRGenCleanup.h"
#include "CIRGenFunction.h"

using namespace clang;
using namespace clang::CIRGen;
using namespace cir;

//===----------------------------------------------------------------------===//
// CIRGenFunction cleanup related
//===----------------------------------------------------------------------===//

/// Build a unconditional branch to the lexical scope cleanup block
/// or with the labeled blocked if already solved.
///
/// Track on scope basis, goto's we need to fix later.
cir::BrOp CIRGenFunction::emitBranchThroughCleanup(mlir::Location Loc,
                                                   JumpDest Dest) {
  // Remove this once we go for making sure unreachable code is
  // well modeled (or not).
  assert(!cir::MissingFeatures::ehStack());

  // Insert a branch: to the cleanup block (unsolved) or to the already
  // materialized label. Keep track of unsolved goto's.
  assert(Dest.getBlock() && "assumes incoming valid dest");
  auto brOp = builder.create<BrOp>(Loc, Dest.getBlock());

  // Calculate the innermost active normal cleanup.
  EHScopeStack::stable_iterator TopCleanup =
      EHStack.getInnermostActiveNormalCleanup();

  // If we're not in an active normal cleanup scope, or if the
  // destination scope is within the innermost active normal cleanup
  // scope, we don't need to worry about fixups.
  if (TopCleanup == EHStack.stable_end() ||
      TopCleanup.encloses(Dest.getScopeDepth())) { // works for invalid
    // FIXME(cir): should we clear insertion point here?
    return brOp;
  }

  // If we can't resolve the destination cleanup scope, just add this
  // to the current cleanup scope as a branch fixup.
  if (!Dest.getScopeDepth().isValid()) {
    BranchFixup &Fixup = EHStack.addBranchFixup();
    Fixup.destination = Dest.getBlock();
    Fixup.destinationIndex = Dest.getDestIndex();
    Fixup.initialBranch = brOp;
    Fixup.optimisticBranchBlock = nullptr;
    // FIXME(cir): should we clear insertion point here?
    return brOp;
  }

  // Otherwise, thread through all the normal cleanups in scope.
  auto index = builder.getUInt32(Dest.getDestIndex(), Loc);
  assert(!cir::MissingFeatures::cleanupIndexAndBIAdjustment());

  // Add this destination to all the scopes involved.
  EHScopeStack::stable_iterator I = TopCleanup;
  EHScopeStack::stable_iterator E = Dest.getScopeDepth();
  if (E.strictlyEncloses(I)) {
    while (true) {
      EHCleanupScope &Scope = cast<EHCleanupScope>(*EHStack.find(I));
      assert(Scope.isNormalCleanup());
      I = Scope.getEnclosingNormalCleanup();

      // If this is the last cleanup we're propagating through, tell it
      // that there's a resolved jump moving through it.
      if (!E.strictlyEncloses(I)) {
        Scope.addBranchAfter(index, Dest.getBlock());
        break;
      }

      // Otherwise, tell the scope that there's a jump propagating
      // through it.  If this isn't new information, all the rest of
      // the work has been done before.
      if (!Scope.addBranchThrough(Dest.getBlock()))
        break;
    }
  }
  return brOp;
}

/// Emits all the code to cause the given temporary to be cleaned up.
void CIRGenFunction::emitCXXTemporary(const CXXTemporary *Temporary,
                                      QualType TempType, Address Ptr) {
  pushDestroy(NormalAndEHCleanup, Ptr, TempType, destroyCXXObject,
              /*useEHCleanup*/ true);
}

Address CIRGenFunction::createCleanupActiveFlag() {
  mlir::Location loc = currSrcLoc ? *currSrcLoc : builder.getUnknownLoc();

  // Create a variable to decide whether the cleanup needs to be run.
  // FIXME: set the insertion point for the alloca to be at the entry
  // basic block of the previous scope, not the entry block of the function.
  Address active = CreateTempAllocaWithoutCast(
      builder.getBoolTy(), CharUnits::One(), loc, "cleanup.cond");
  mlir::Value falseVal, trueVal;
  {
    // Place true/false flags close to their allocas.
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointAfterValue(active.getPointer());
    falseVal = builder.getFalse(loc);
    trueVal = builder.getTrue(loc);
  }

  // Initialize it to false at a site that's guaranteed to be run
  // before each evaluation.
  setBeforeOutermostConditional(falseVal, active);

  // Initialize it to true at the current location.
  builder.createStore(loc, trueVal, active);
  return active;
}

DominatingValue<RValue>::saved_type
DominatingValue<RValue>::saved_type::save(CIRGenFunction &cgf, RValue rv) {
  if (rv.isScalar()) {
    mlir::Value val = rv.getScalarVal();
    return saved_type(DominatingCIRValue::save(cgf, val),
                      DominatingCIRValue::needsSaving(val) ? ScalarAddress
                                                           : ScalarLiteral);
  }

  if (rv.isComplex()) {
    llvm_unreachable("complex NYI");
  }

  llvm_unreachable("aggregate NYI");
}

/// Given a saved r-value produced by SaveRValue, perform the code
/// necessary to restore it to usability at the current insertion
/// point.
RValue DominatingValue<RValue>::saved_type::restore(CIRGenFunction &CGF) {
  switch (K) {
  case ScalarLiteral:
  case ScalarAddress:
    return RValue::get(DominatingCIRValue::restore(CGF, Vals.first));
  case AggregateLiteral:
  case AggregateAddress:
    return RValue::getAggregate(
        DominatingValue<Address>::restore(CGF, AggregateAddr));
  case ComplexAddress: {
    llvm_unreachable("NYI");
  }
  }

  llvm_unreachable("bad saved r-value kind");
}

static bool IsUsedAsEHCleanup(EHScopeStack &EHStack,
                              EHScopeStack::stable_iterator cleanup) {
  // If we needed an EH block for any reason, that counts.
  if (EHStack.find(cleanup)->hasEHBranches())
    return true;

  // Check whether any enclosed cleanups were needed.
  for (EHScopeStack::stable_iterator i = EHStack.getInnermostEHScope();
       i != cleanup;) {
    assert(cleanup.strictlyEncloses(i));

    EHScope &scope = *EHStack.find(i);
    if (scope.hasEHBranches())
      return true;

    i = scope.getEnclosingEHScope();
  }

  return false;
}

enum ForActivation_t { ForActivation, ForDeactivation };

/// The given cleanup block is changing activation state.  Configure a
/// cleanup variable if necessary.
///
/// It would be good if we had some way of determining if there were
/// extra uses *after* the change-over point.
static void setupCleanupBlockActivation(CIRGenFunction &CGF,
                                        EHScopeStack::stable_iterator C,
                                        ForActivation_t kind,
                                        mlir::Operation *dominatingIP) {
  EHCleanupScope &Scope = cast<EHCleanupScope>(*CGF.EHStack.find(C));

  // We always need the flag if we're activating the cleanup in a
  // conditional context, because we have to assume that the current
  // location doesn't necessarily dominate the cleanup's code.
  bool isActivatedInConditional =
      (kind == ForActivation && CGF.isInConditionalBranch());

  bool needFlag = false;

  // Calculate whether the cleanup was used:

  //   - as a normal cleanup
  if (Scope.isNormalCleanup()) {
    Scope.setTestFlagInNormalCleanup();
    needFlag = true;
  }

  //  - as an EH cleanup
  if (Scope.isEHCleanup() &&
      (isActivatedInConditional || IsUsedAsEHCleanup(CGF.EHStack, C))) {
    Scope.setTestFlagInEHCleanup();
    needFlag = true;
  }

  // If it hasn't yet been used as either, we're done.
  if (!needFlag)
    return;

  Address var = Scope.getActiveFlag();
  if (!var.isValid()) {
    llvm_unreachable("NYI");
  }

  auto builder = CGF.getBuilder();
  mlir::Location loc = var.getPointer().getLoc();
  mlir::Value trueOrFalse =
      kind == ForActivation ? builder.getTrue(loc) : builder.getFalse(loc);
  CGF.getBuilder().createStore(loc, trueOrFalse, var);
}

/// Deactive a cleanup that was created in an active state.
void CIRGenFunction::DeactivateCleanupBlock(EHScopeStack::stable_iterator C,
                                            mlir::Operation *dominatingIP) {
  assert(C != EHStack.stable_end() && "deactivating bottom of stack?");
  EHCleanupScope &Scope = cast<EHCleanupScope>(*EHStack.find(C));
  assert(Scope.isActive() && "double deactivation");

  // If it's the top of the stack, just pop it, but do so only if it belongs
  // to the current RunCleanupsScope.
  if (C == EHStack.stable_begin() &&
      CurrentCleanupScopeDepth.strictlyEncloses(C)) {
    // Per comment below, checking EHAsynch is not really necessary
    // it's there to assure zero-impact w/o EHAsynch option
    if (!Scope.isNormalCleanup() && getLangOpts().EHAsynch) {
      llvm_unreachable("NYI");
    } else {
      // From LLVM: If it's a normal cleanup, we need to pretend that the
      // fallthrough is unreachable.
      // CIR remarks: LLVM uses an empty insertion point to signal behavior
      // change to other codegen paths (triggered by PopCleanupBlock).
      // CIRGen doesn't do that yet, but let's mimic just in case.
      mlir::OpBuilder::InsertionGuard guard(builder);
      builder.clearInsertionPoint();
      PopCleanupBlock();
    }
    return;
  }

  // Otherwise, follow the general case.
  setupCleanupBlockActivation(*this, C, ForDeactivation, dominatingIP);
  Scope.setActive(false);
}

void CIRGenFunction::initFullExprCleanupWithFlag(Address ActiveFlag) {
  // Set that as the active flag in the cleanup.
  EHCleanupScope &cleanup = cast<EHCleanupScope>(*EHStack.begin());
  assert(!cleanup.hasActiveFlag() && "cleanup already has active flag?");
  cleanup.setActiveFlag(ActiveFlag);

  if (cleanup.isNormalCleanup())
    cleanup.setTestFlagInNormalCleanup();
  if (cleanup.isEHCleanup())
    cleanup.setTestFlagInEHCleanup();
}

/// We don't need a normal entry block for the given cleanup.
/// Optimistic fixup branches can cause these blocks to come into
/// existence anyway;  if so, destroy it.
///
/// The validity of this transformation is very much specific to the
/// exact ways in which we form branches to cleanup entries.
static void destroyOptimisticNormalEntry(CIRGenFunction &CGF,
                                         EHCleanupScope &scope) {
  auto *entry = scope.getNormalBlock();
  if (!entry)
    return;

  llvm_unreachable("NYI");
}

static void emitCleanup(CIRGenFunction &CGF, EHScopeStack::Cleanup *Fn,
                        EHScopeStack::Cleanup::Flags flags,
                        Address ActiveFlag) {
  auto emitCleanup = [&]() {
    // Ask the cleanup to emit itself.
    assert(CGF.HaveInsertPoint() && "expected insertion point");
    Fn->Emit(CGF, flags);
    assert(CGF.HaveInsertPoint() && "cleanup ended with no insertion point?");
  };

  // If there's an active flag, load it and skip the cleanup if it's
  // false.
  CIRGenBuilderTy &builder = CGF.getBuilder();
  mlir::Location loc =
      CGF.currSrcLoc ? *CGF.currSrcLoc : builder.getUnknownLoc();

  if (ActiveFlag.isValid()) {
    mlir::Value isActive = builder.createLoad(loc, ActiveFlag);
    builder.create<cir::IfOp>(loc, isActive, false,
                              [&](mlir::OpBuilder &b, mlir::Location) {
                                emitCleanup();
                                builder.createYield(loc);
                              });
  } else {
    emitCleanup();
  }
  // No need to emit continuation block because CIR uses a cir.if.
}

static mlir::Block *createNormalEntry(CIRGenFunction &cgf,
                                      EHCleanupScope &scope) {
  assert(scope.isNormalCleanup());
  mlir::Block *entry = scope.getNormalBlock();
  if (!entry) {
    mlir::OpBuilder::InsertionGuard guard(cgf.getBuilder());
    entry = cgf.currLexScope->getOrCreateCleanupBlock(cgf.getBuilder());
    scope.setNormalBlock(entry);
  }
  return entry;
}

/// Pops a cleanup block. If the block includes a normal cleanup, the
/// current insertion point is threaded through the cleanup, as are
/// any branch fixups on the cleanup.
void CIRGenFunction::PopCleanupBlock(bool FallthroughIsBranchThrough) {
  assert(!EHStack.empty() && "cleanup stack is empty!");
  assert(isa<EHCleanupScope>(*EHStack.begin()) && "top not a cleanup!");
  EHCleanupScope &Scope = cast<EHCleanupScope>(*EHStack.begin());
  assert(Scope.getFixupDepth() <= EHStack.getNumBranchFixups());

  // Remember activation information.
  bool IsActive = Scope.isActive();
  Address NormalActiveFlag = Scope.shouldTestFlagInNormalCleanup()
                                 ? Scope.getActiveFlag()
                                 : Address::invalid();
  Address EHActiveFlag = Scope.shouldTestFlagInEHCleanup()
                             ? Scope.getActiveFlag()
                             : Address::invalid();

  // Check whether we need an EH cleanup. This is only true if we've
  // generated a lazy EH cleanup block.
  auto *ehEntry = Scope.getCachedEHDispatchBlock();
  assert(Scope.hasEHBranches() == (ehEntry != nullptr));
  bool RequiresEHCleanup = (ehEntry != nullptr);
  EHScopeStack::stable_iterator EHParent = Scope.getEnclosingEHScope();

  // Check the three conditions which might require a normal cleanup:

  // - whether there are branch fix-ups through this cleanup
  unsigned FixupDepth = Scope.getFixupDepth();
  bool HasFixups = EHStack.getNumBranchFixups() != FixupDepth;

  // - whether there are branch-throughs or branch-afters
  bool HasExistingBranches = Scope.hasBranches();

  // - whether there's a fallthrough
  auto *FallthroughSource = builder.getInsertionBlock();
  bool HasFallthrough =
      (FallthroughSource != nullptr && (IsActive || HasExistingBranches));

  // Branch-through fall-throughs leave the insertion point set to the
  // end of the last cleanup, which points to the current scope.  The
  // rest of CIR gen doesn't need to worry about this; it only happens
  // during the execution of PopCleanupBlocks().
  bool HasTerminator = FallthroughSource &&
                       FallthroughSource->mightHaveTerminator() &&
                       FallthroughSource->getTerminator();
  bool HasPrebranchedFallthrough =
      HasTerminator && !isa<cir::YieldOp>(FallthroughSource->getTerminator());

  // If this is a normal cleanup, then having a prebranched
  // fallthrough implies that the fallthrough source unconditionally
  // jumps here.
  assert(!Scope.isNormalCleanup() || !HasPrebranchedFallthrough ||
         (Scope.getNormalBlock() &&
          FallthroughSource->getTerminator()->getSuccessor(0) ==
              Scope.getNormalBlock()));

  bool RequiresNormalCleanup = false;
  if (Scope.isNormalCleanup() &&
      (HasFixups || HasExistingBranches || HasFallthrough)) {
    RequiresNormalCleanup = true;
  }

  // If we have a prebranched fallthrough into an inactive normal
  // cleanup, rewrite it so that it leads to the appropriate place.
  if (Scope.isNormalCleanup() && HasPrebranchedFallthrough && !IsActive) {
    llvm_unreachable("NYI");
  }

  // If we don't need the cleanup at all, we're done.
  if (!RequiresNormalCleanup && !RequiresEHCleanup) {
    destroyOptimisticNormalEntry(*this, Scope);
    EHStack.popCleanup(); // safe because there are no fixups
    assert(EHStack.getNumBranchFixups() == 0 || EHStack.hasNormalCleanups());
    return;
  }

  // Copy the cleanup emission data out.  This uses either a stack
  // array or malloc'd memory, depending on the size, which is
  // behavior that SmallVector would provide, if we could use it
  // here. Unfortunately, if you ask for a SmallVector<char>, the
  // alignment isn't sufficient.
  auto *CleanupSource = reinterpret_cast<char *>(Scope.getCleanupBuffer());
  alignas(EHScopeStack::ScopeStackAlignment) char
      CleanupBufferStack[8 * sizeof(void *)];
  std::unique_ptr<char[]> CleanupBufferHeap;
  size_t CleanupSize = Scope.getCleanupSize();
  EHScopeStack::Cleanup *Fn;

  if (CleanupSize <= sizeof(CleanupBufferStack)) {
    memcpy(CleanupBufferStack, CleanupSource, CleanupSize);
    Fn = reinterpret_cast<EHScopeStack::Cleanup *>(CleanupBufferStack);
  } else {
    CleanupBufferHeap.reset(new char[CleanupSize]);
    memcpy(CleanupBufferHeap.get(), CleanupSource, CleanupSize);
    Fn = reinterpret_cast<EHScopeStack::Cleanup *>(CleanupBufferHeap.get());
  }

  EHScopeStack::Cleanup::Flags cleanupFlags;
  if (Scope.isNormalCleanup())
    cleanupFlags.setIsNormalCleanupKind();
  if (Scope.isEHCleanup())
    cleanupFlags.setIsEHCleanupKind();

  // Under -EHa, invoke seh.scope.end() to mark scope end before dtor
  bool IsEHa = getLangOpts().EHAsynch && !Scope.isLifetimeMarker();
  // const EHPersonality &Personality = EHPersonality::get(*this);
  if (!RequiresNormalCleanup) {
    // Mark CPP scope end for passed-by-value Arg temp
    //   per Windows ABI which is "normally" Cleanup in callee
    if (IsEHa && isInvokeDest()) {
      // If we are deactivating a normal cleanup then we don't have a
      // fallthrough. Restore original IP to emit CPP scope ends in the correct
      // block.
      llvm_unreachable("NYI");
    }
    destroyOptimisticNormalEntry(*this, Scope);
    Scope.markEmitted();
    EHStack.popCleanup();
  } else {
    // If we have a fallthrough and no other need for the cleanup,
    // emit it directly.
    if (HasFallthrough && !HasPrebranchedFallthrough && !HasFixups &&
        !HasExistingBranches) {

      // mark SEH scope end for fall-through flow
      if (IsEHa) {
        llvm_unreachable("NYI");
      }

      destroyOptimisticNormalEntry(*this, Scope);
      EHStack.popCleanup();
      Scope.markEmitted();
      emitCleanup(*this, Fn, cleanupFlags, NormalActiveFlag);

      // Otherwise, the best approach is to thread everything through
      // the cleanup block and then try to clean up after ourselves.
    } else {
      // Force the entry block to exist.
      mlir::Block *normalEntry = createNormalEntry(*this, Scope);

      // I.  Set up the fallthrough edge in.
      mlir::OpBuilder::InsertPoint savedInactiveFallthroughIP;

      // If there's a fallthrough, we need to store the cleanup
      // destination index. For fall-throughs this is always zero.
      if (HasFallthrough) {
        if (!HasPrebranchedFallthrough) {
          assert(!cir::MissingFeatures::cleanupDestinationIndex());
        }

        // Otherwise, save and clear the IP if we don't have fallthrough
        // because the cleanup is inactive.
      } else if (FallthroughSource) {
        assert(!IsActive && "source without fallthrough for active cleanup");
        savedInactiveFallthroughIP = getBuilder().saveInsertionPoint();
      }

      // II.  Emit the entry block.  This implicitly branches to it if
      // we have fallthrough.  All the fixups and existing branches
      // should already be branched to it.
      builder.setInsertionPointToEnd(normalEntry);

      // intercept normal cleanup to mark SEH scope end
      if (IsEHa) {
        llvm_unreachable("NYI");
      }

      // III.  Figure out where we're going and build the cleanup
      // epilogue.
      bool HasEnclosingCleanups =
          (Scope.getEnclosingNormalCleanup() != EHStack.stable_end());

      // Compute the branch-through dest if we need it:
      //   - if there are branch-throughs threaded through the scope
      //   - if fall-through is a branch-through
      //   - if there are fixups that will be optimistically forwarded
      //     to the enclosing cleanup
      mlir::Block *branchThroughDest = nullptr;
      if (Scope.hasBranchThroughs() ||
          (FallthroughSource && FallthroughIsBranchThrough) ||
          (HasFixups && HasEnclosingCleanups)) {
        llvm_unreachable("NYI");
      }

      mlir::Block *fallthroughDest = nullptr;

      // If there's exactly one branch-after and no other threads,
      // we can route it without a switch.
      // Skip for SEH, since ExitSwitch is used to generate code to indicate
      // abnormal termination. (SEH: Except _leave and fall-through at
      // the end, all other exits in a _try (return/goto/continue/break)
      // are considered as abnormal terminations, using NormalCleanupDestSlot
      // to indicate abnormal termination)
      if (!Scope.hasBranchThroughs() && !HasFixups && !HasFallthrough &&
          !currentFunctionUsesSEHTry() && Scope.getNumBranchAfters() == 1) {
        llvm_unreachable("NYI");
        // Build a switch-out if we need it:
        //   - if there are branch-afters threaded through the scope
        //   - if fall-through is a branch-after
        //   - if there are fixups that have nowhere left to go and
        //     so must be immediately resolved
      } else if (Scope.getNumBranchAfters() ||
                 (HasFallthrough && !FallthroughIsBranchThrough) ||
                 (HasFixups && !HasEnclosingCleanups)) {
        assert(!cir::MissingFeatures::cleanupBranchAfterSwitch());
      } else {
        // We should always have a branch-through destination in this case.
        assert(branchThroughDest);
        assert(!cir::MissingFeatures::cleanupAlwaysBranchThrough());
      }

      // IV.  Pop the cleanup and emit it.
      Scope.markEmitted();
      EHStack.popCleanup();
      assert(EHStack.hasNormalCleanups() == HasEnclosingCleanups);

      emitCleanup(*this, Fn, cleanupFlags, NormalActiveFlag);

      // Append the prepared cleanup prologue from above.
      assert(!cir::MissingFeatures::cleanupAppendInsts());

      // Optimistically hope that any fixups will continue falling through.
      for (unsigned I = FixupDepth, E = EHStack.getNumBranchFixups(); I < E;
           ++I) {
        llvm_unreachable("NYI");
      }

      // V.  Set up the fallthrough edge out.

      // Case 1: a fallthrough source exists but doesn't branch to the
      // cleanup because the cleanup is inactive.
      if (!HasFallthrough && FallthroughSource) {
        // Prebranched fallthrough was forwarded earlier.
        // Non-prebranched fallthrough doesn't need to be forwarded.
        // Either way, all we need to do is restore the IP we cleared before.
        assert(!IsActive);
        llvm_unreachable("NYI");

        // Case 2: a fallthrough source exists and should branch to the
        // cleanup, but we're not supposed to branch through to the next
        // cleanup.
      } else if (HasFallthrough && fallthroughDest) {
        llvm_unreachable("NYI");

        // Case 3: a fallthrough source exists and should branch to the
        // cleanup and then through to the next.
      } else if (HasFallthrough) {
        // Everything is already set up for this.

        // Case 4: no fallthrough source exists.
      } else {
        // FIXME(cir): should we clear insertion point here?
      }

      // VI.  Assorted cleaning.

      // Check whether we can merge NormalEntry into a single predecessor.
      // This might invalidate (non-IR) pointers to NormalEntry.
      //
      // If it did invalidate those pointers, and NormalEntry was the same
      // as NormalExit, go back and patch up the fixups.
      assert(!cir::MissingFeatures::simplifyCleanupEntry());
    }
  }

  assert(EHStack.hasNormalCleanups() || EHStack.getNumBranchFixups() == 0);

  // Emit the EH cleanup if required.
  if (RequiresEHCleanup) {
    cir::TryOp tryOp = ehEntry->getParentOp()->getParentOfType<cir::TryOp>();
    auto *nextAction = getEHDispatchBlock(EHParent, tryOp);
    (void)nextAction;

    // Push a terminate scope or cleanupendpad scope around the potentially
    // throwing cleanups. For funclet EH personalities, the cleanupendpad models
    // program termination when cleanups throw.
    bool PushedTerminate = false;
    SaveAndRestore RestoreCurrentFuncletPad(CurrentFuncletPad);
    mlir::Operation *CPI = nullptr;

    const EHPersonality &Personality = EHPersonality::get(*this);
    if (Personality.usesFuncletPads()) {
      llvm_unreachable("NYI");
    }

    // Non-MSVC personalities need to terminate when an EH cleanup throws.
    if (!Personality.isMSVCPersonality()) {
      EHStack.pushTerminate();
      PushedTerminate = true;
    } else if (IsEHa && isInvokeDest()) {
      llvm_unreachable("NYI");
    }

    // We only actually emit the cleanup code if the cleanup is either
    // active or was used before it was deactivated.
    if (EHActiveFlag.isValid() || IsActive) {
      cleanupFlags.setIsForEHCleanup();
      mlir::OpBuilder::InsertionGuard guard(builder);

      auto yield = cast<YieldOp>(ehEntry->getTerminator());
      builder.setInsertionPoint(yield);
      emitCleanup(*this, Fn, cleanupFlags, EHActiveFlag);
    }

    if (CPI)
      llvm_unreachable("NYI");
    else {
      // In LLVM traditional codegen, here's where it branches off to
      // nextAction. CIR does not have a flat layout at this point, so
      // instead patch all the landing pads that need to run this cleanup
      // as well.
      mlir::Block *currBlock = ehEntry;
      while (currBlock && cleanupsToPatch.contains(currBlock)) {
        mlir::OpBuilder::InsertionGuard guard(builder);
        mlir::Block *blockToPatch = cleanupsToPatch[currBlock];
        auto currYield = cast<YieldOp>(blockToPatch->getTerminator());
        builder.setInsertionPoint(currYield);

        // If nextAction is an EH resume block, also update all try locations
        // for these "to-patch" blocks with the appropriate resume content.
        if (nextAction == ehResumeBlock) {
          if (auto tryToPatch =
                  currYield->getParentOp()->getParentOfType<cir::TryOp>()) {
            if (!tryToPatch.getSynthetic()) {
              mlir::Block *resumeBlockToPatch =
                  tryToPatch.getCatchUnwindEntryBlock();
              emitEHResumeBlock(/*isCleanup=*/true, resumeBlockToPatch,
                                tryToPatch.getLoc());
            }
          }
        }

        emitCleanup(*this, Fn, cleanupFlags, EHActiveFlag);
        currBlock = blockToPatch;
      }

      // The nextAction is yet to be populated, register that this
      // cleanup should also incorporate any cleanup from nextAction
      // when available.
      cleanupsToPatch[nextAction] = ehEntry;
    }

    // Leave the terminate scope.
    if (PushedTerminate)
      EHStack.popTerminate();

    // FIXME(cir): LLVM traditional codegen tries to simplify some of the
    // codegen here. Once we are further down with EH support revisit whether we
    // need to this during lowering.
    assert(!cir::MissingFeatures::simplifyCleanupEntry());
  }
}

/// Pops cleanup blocks until the given savepoint is reached.
void CIRGenFunction::PopCleanupBlocks(
    EHScopeStack::stable_iterator Old,
    std::initializer_list<mlir::Value *> ValuesToReload) {
  assert(Old.isValid());

  bool HadBranches = false;
  while (EHStack.stable_begin() != Old) {
    EHCleanupScope &Scope = cast<EHCleanupScope>(*EHStack.begin());
    HadBranches |= Scope.hasBranches();

    // As long as Old strictly encloses the scope's enclosing normal
    // cleanup, we're going to emit another normal cleanup which
    // fallthrough can propagate through.
    bool FallThroughIsBranchThrough =
        Old.strictlyEncloses(Scope.getEnclosingNormalCleanup());

    PopCleanupBlock(FallThroughIsBranchThrough);
  }

  // If we didn't have any branches, the insertion point before cleanups must
  // dominate the current insertion point and we don't need to reload any
  // values.
  if (!HadBranches)
    return;

  llvm_unreachable("NYI");
}

/// Pops cleanup blocks until the given savepoint is reached, then add the
/// cleanups from the given savepoint in the lifetime-extended cleanups stack.
void CIRGenFunction::PopCleanupBlocks(
    EHScopeStack::stable_iterator Old, size_t OldLifetimeExtendedSize,
    std::initializer_list<mlir::Value *> ValuesToReload) {
  PopCleanupBlocks(Old, ValuesToReload);

  // Move our deferred cleanups onto the EH stack.
  for (size_t I = OldLifetimeExtendedSize,
              E = LifetimeExtendedCleanupStack.size();
       I != E;
       /**/) {
    // Alignment should be guaranteed by the vptrs in the individual cleanups.
    assert((I % alignof(LifetimeExtendedCleanupHeader) == 0) &&
           "misaligned cleanup stack entry");

    LifetimeExtendedCleanupHeader &Header =
        reinterpret_cast<LifetimeExtendedCleanupHeader &>(
            LifetimeExtendedCleanupStack[I]);
    I += sizeof(Header);

    EHStack.pushCopyOfCleanup(
        Header.getKind(), &LifetimeExtendedCleanupStack[I], Header.getSize());
    I += Header.getSize();

    if (Header.isConditional()) {
      Address ActiveFlag =
          reinterpret_cast<Address &>(LifetimeExtendedCleanupStack[I]);
      initFullExprCleanupWithFlag(ActiveFlag);
      I += sizeof(ActiveFlag);
    }
  }
  LifetimeExtendedCleanupStack.resize(OldLifetimeExtendedSize);
}

//===----------------------------------------------------------------------===//
// EHScopeStack
//===----------------------------------------------------------------------===//

void EHScopeStack::Cleanup::anchor() {}

EHScopeStack::stable_iterator
EHScopeStack::getInnermostActiveNormalCleanup() const {
  for (stable_iterator si = getInnermostNormalCleanup(), se = stable_end();
       si != se;) {
    EHCleanupScope &cleanup = cast<EHCleanupScope>(*find(si));
    if (cleanup.isActive())
      return si;
    si = cleanup.getEnclosingNormalCleanup();
  }
  return stable_end();
}

/// Push an entry of the given size onto this protected-scope stack.
char *EHScopeStack::allocate(size_t Size) {
  Size = llvm::alignTo(Size, ScopeStackAlignment);
  if (!StartOfBuffer) {
    unsigned Capacity = 1024;
    while (Capacity < Size)
      Capacity *= 2;
    StartOfBuffer = new char[Capacity];
    StartOfData = EndOfBuffer = StartOfBuffer + Capacity;
  } else if (static_cast<size_t>(StartOfData - StartOfBuffer) < Size) {
    unsigned CurrentCapacity = EndOfBuffer - StartOfBuffer;
    unsigned UsedCapacity = CurrentCapacity - (StartOfData - StartOfBuffer);

    unsigned NewCapacity = CurrentCapacity;
    do {
      NewCapacity *= 2;
    } while (NewCapacity < UsedCapacity + Size);

    char *NewStartOfBuffer = new char[NewCapacity];
    char *NewEndOfBuffer = NewStartOfBuffer + NewCapacity;
    char *NewStartOfData = NewEndOfBuffer - UsedCapacity;
    memcpy(NewStartOfData, StartOfData, UsedCapacity);
    delete[] StartOfBuffer;
    StartOfBuffer = NewStartOfBuffer;
    EndOfBuffer = NewEndOfBuffer;
    StartOfData = NewStartOfData;
  }

  assert(StartOfBuffer + Size <= StartOfData);
  StartOfData -= Size;
  return StartOfData;
}

void *EHScopeStack::pushCleanup(CleanupKind Kind, size_t Size) {
  char *Buffer = allocate(EHCleanupScope::getSizeForCleanupSize(Size));
  bool IsNormalCleanup = Kind & NormalCleanup;
  bool IsEHCleanup = Kind & EHCleanup;
  bool IsLifetimeMarker = Kind & LifetimeMarker;

  // Per C++ [except.terminate], it is implementation-defined whether none,
  // some, or all cleanups are called before std::terminate. Thus, when
  // terminate is the current EH scope, we may skip adding any EH cleanup
  // scopes.
  if (InnermostEHScope != stable_end() &&
      find(InnermostEHScope)->getKind() == EHScope::Terminate)
    IsEHCleanup = false;

  EHCleanupScope *Scope = new (Buffer)
      EHCleanupScope(IsNormalCleanup, IsEHCleanup, Size, BranchFixups.size(),
                     InnermostNormalCleanup, InnermostEHScope);
  if (IsNormalCleanup)
    InnermostNormalCleanup = stable_begin();
  if (IsEHCleanup)
    InnermostEHScope = stable_begin();
  if (IsLifetimeMarker)
    llvm_unreachable("NYI");

  // With Windows -EHa, Invoke llvm.seh.scope.begin() for EHCleanup
  if (CGF->getLangOpts().EHAsynch && IsEHCleanup && !IsLifetimeMarker &&
      CGF->getTarget().getCXXABI().isMicrosoft())
    llvm_unreachable("NYI");

  return Scope->getCleanupBuffer();
}

void EHScopeStack::popCleanup() {
  assert(!empty() && "popping exception stack when not empty");

  assert(isa<EHCleanupScope>(*begin()));
  EHCleanupScope &Cleanup = cast<EHCleanupScope>(*begin());
  InnermostNormalCleanup = Cleanup.getEnclosingNormalCleanup();
  InnermostEHScope = Cleanup.getEnclosingEHScope();
  deallocate(Cleanup.getAllocatedSize());

  // Destroy the cleanup.
  Cleanup.Destroy();

  // Check whether we can shrink the branch-fixups stack.
  if (!BranchFixups.empty()) {
    // If we no longer have any normal cleanups, all the fixups are
    // complete.
    if (!hasNormalCleanups())
      BranchFixups.clear();

    // Otherwise we can still trim out unnecessary nulls.
    else
      popNullFixups();
  }
}

void EHScopeStack::deallocate(size_t Size) {
  StartOfData += llvm::alignTo(Size, ScopeStackAlignment);
}

/// Remove any 'null' fixups on the stack.  However, we can't pop more
/// fixups than the fixup depth on the innermost normal cleanup, or
/// else fixups that we try to add to that cleanup will end up in the
/// wrong place.  We *could* try to shrink fixup depths, but that's
/// actually a lot of work for little benefit.
void EHScopeStack::popNullFixups() {
  // We expect this to only be called when there's still an innermost
  // normal cleanup;  otherwise there really shouldn't be any fixups.
  llvm_unreachable("NYI");
}

bool EHScopeStack::requiresLandingPad() const {
  for (stable_iterator si = getInnermostEHScope(); si != stable_end();) {
    // Skip lifetime markers.
    if (auto *cleanup = dyn_cast<EHCleanupScope>(&*find(si)))
      if (cleanup->isLifetimeMarker()) {
        si = cleanup->getEnclosingEHScope();
        continue;
      }
    return true;
  }

  return false;
}

EHCatchScope *EHScopeStack::pushCatch(unsigned numHandlers) {
  char *buffer = allocate(EHCatchScope::getSizeForNumHandlers(numHandlers));
  EHCatchScope *scope =
      new (buffer) EHCatchScope(numHandlers, InnermostEHScope);
  InnermostEHScope = stable_begin();
  return scope;
}

void EHScopeStack::pushTerminate() {
  char *Buffer = allocate(EHTerminateScope::getSize());
  new (Buffer) EHTerminateScope(InnermostEHScope);
  InnermostEHScope = stable_begin();
}

bool EHScopeStack::containsOnlyLifetimeMarkers(
    EHScopeStack::stable_iterator old) const {
  for (EHScopeStack::iterator it = begin(); stabilize(it) != old; it++) {
    EHCleanupScope *cleanup = dyn_cast<EHCleanupScope>(&*it);
    if (!cleanup || !cleanup->isLifetimeMarker())
      return false;
  }

  return true;
}
