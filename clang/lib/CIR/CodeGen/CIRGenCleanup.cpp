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

#include "CIRGenCleanup.h"
#include "CIRGenFunction.h"

using namespace cir;
using namespace clang;
using namespace mlir::cir;

//===----------------------------------------------------------------------===//
// CIRGenFunction cleanup related
//===----------------------------------------------------------------------===//

/// Build a unconditional branch to the lexical scope cleanup block
/// or with the labeled blocked if already solved.
///
/// Track on scope basis, goto's we need to fix later.
mlir::cir::BrOp CIRGenFunction::buildBranchThroughCleanup(mlir::Location Loc,
                                                          JumpDest Dest) {
  // Remove this once we go for making sure unreachable code is
  // well modeled (or not).
  assert(builder.getInsertionBlock() && "not yet implemented");
  assert(!UnimplementedFeature::ehStack());

  // Insert a branch: to the cleanup block (unsolved) or to the already
  // materialized label. Keep track of unsolved goto's.
  return builder.create<BrOp>(Loc, Dest.isValid() ? Dest.getBlock()
                                                  : ReturnBlock().getBlock());
}

/// Emits all the code to cause the given temporary to be cleaned up.
void CIRGenFunction::buildCXXTemporary(const CXXTemporary *Temporary,
                                       QualType TempType, Address Ptr) {
  pushDestroy(NormalAndEHCleanup, Ptr, TempType, destroyCXXObject,
              /*useEHCleanup*/ true);
}

Address CIRGenFunction::createCleanupActiveFlag() { llvm_unreachable("NYI"); }

DominatingValue<RValue>::saved_type
DominatingValue<RValue>::saved_type::save(CIRGenFunction &CGF, RValue rv) {
  llvm_unreachable("NYI");
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

  llvm_unreachable("NYI");
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

static void buildCleanup(CIRGenFunction &CGF, EHScopeStack::Cleanup *Fn,
                         EHScopeStack::Cleanup::Flags flags,
                         Address ActiveFlag) {
  // If there's an active flag, load it and skip the cleanup if it's
  // false.
  if (ActiveFlag.isValid()) {
    llvm_unreachable("NYI");
  }

  // Ask the cleanup to emit itself.
  Fn->Emit(CGF, flags);
  assert(CGF.HaveInsertPoint() && "cleanup ended with no insertion point?");

  // Emit the continuation block if there was an active flag.
  if (ActiveFlag.isValid()) {
    llvm_unreachable("NYI");
  }
}

/// Pops a cleanup block. If the block includes a normal cleanup, the
/// current insertion point is threaded through the cleanup, as are
/// any branch fixups on the cleanup.
void CIRGenFunction::PopCleanupBlock(bool FallthroughIsBranchThrough) {
  assert(!EHStack.empty() && "cleanup stack is empty!");
  assert(isa<EHCleanupScope>(*EHStack.begin()) && "top not a cleanup!");
  [[maybe_unused]] EHCleanupScope &Scope =
      cast<EHCleanupScope>(*EHStack.begin());
  assert(Scope.getFixupDepth() <= EHStack.getNumBranchFixups());

  // Remember activation information.
  bool IsActive = Scope.isActive();
  Address NormalActiveFlag = Scope.shouldTestFlagInNormalCleanup()
                                 ? Scope.getActiveFlag()
                                 : Address::invalid();
  [[maybe_unused]] Address EHActiveFlag = Scope.shouldTestFlagInEHCleanup()
                                              ? Scope.getActiveFlag()
                                              : Address::invalid();

  // Check whether we need an EH cleanup. This is only true if we've
  // generated a lazy EH cleanup block.
  auto *EHEntry = Scope.getCachedEHDispatchBlock();
  assert(Scope.hasEHBranches() == (EHEntry != nullptr));
  bool RequiresEHCleanup = (EHEntry != nullptr);

  // Check the three conditions which might require a normal cleanup:

  // - whether there are branch fix-ups through this cleanup
  unsigned FixupDepth = Scope.getFixupDepth();
  bool HasFixups = EHStack.getNumBranchFixups() != FixupDepth;

  // - whether there are branch-throughs or branch-afters
  bool HasExistingBranches = Scope.hasBranches();

  // - whether there's a fallthrough
  auto *FallthroughSource = builder.getInsertionBlock();
  bool HasFallthrough = (FallthroughSource != nullptr && IsActive);

  // Branch-through fall-throughs leave the insertion point set to the
  // end of the last cleanup, which points to the current scope.  The
  // rest of CIR gen doesn't need to worry about this; it only happens
  // during the execution of PopCleanupBlocks().
  bool HasTerminator = FallthroughSource &&
                       FallthroughSource->mightHaveTerminator() &&
                       FallthroughSource->getTerminator();
  bool HasPrebranchedFallthrough =
      HasTerminator &&
      !isa<mlir::cir::YieldOp>(FallthroughSource->getTerminator());

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
    llvm_unreachable("NYI");
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
      buildCleanup(*this, Fn, cleanupFlags, NormalActiveFlag);

      // Otherwise, the best approach is to thread everything through
      // the cleanup block and then try to clean up after ourselves.
    } else {
      llvm_unreachable("NYI");
    }
  }

  assert(EHStack.hasNormalCleanups() || EHStack.getNumBranchFixups() == 0);

  // Emit the EH cleanup if required.
  if (RequiresEHCleanup) {
    llvm_unreachable("NYI");
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
