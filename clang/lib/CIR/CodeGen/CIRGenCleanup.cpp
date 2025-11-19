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

#include "clang/CIR/MissingFeatures.h"

using namespace clang;
using namespace clang::CIRGen;

//===----------------------------------------------------------------------===//
// CIRGenFunction cleanup related
//===----------------------------------------------------------------------===//

/// Build a unconditional branch to the lexical scope cleanup block
/// or with the labeled blocked if already solved.
///
/// Track on scope basis, goto's we need to fix later.
cir::BrOp CIRGenFunction::emitBranchThroughCleanup(mlir::Location loc,
                                                   JumpDest dest) {
  // Insert a branch: to the cleanup block (unsolved) or to the already
  // materialized label. Keep track of unsolved goto's.
  assert(dest.getBlock() && "assumes incoming valid dest");
  auto brOp = cir::BrOp::create(builder, loc, dest.getBlock());

  // Calculate the innermost active normal cleanup.
  EHScopeStack::stable_iterator topCleanup =
      ehStack.getInnermostActiveNormalCleanup();

  // If we're not in an active normal cleanup scope, or if the
  // destination scope is within the innermost active normal cleanup
  // scope, we don't need to worry about fixups.
  if (topCleanup == ehStack.stable_end() ||
      topCleanup.encloses(dest.getScopeDepth())) { // works for invalid
    // FIXME(cir): should we clear insertion point here?
    return brOp;
  }

  // If we can't resolve the destination cleanup scope, just add this
  // to the current cleanup scope as a branch fixup.
  if (!dest.getScopeDepth().isValid()) {
    BranchFixup &fixup = ehStack.addBranchFixup();
    fixup.destination = dest.getBlock();
    fixup.destinationIndex = dest.getDestIndex();
    fixup.initialBranch = brOp;
    fixup.optimisticBranchBlock = nullptr;
    // FIXME(cir): should we clear insertion point here?
    return brOp;
  }

  cgm.errorNYI(loc, "emitBranchThroughCleanup: valid destination scope depth");
  return brOp;
}

/// Emits all the code to cause the given temporary to be cleaned up.
void CIRGenFunction::emitCXXTemporary(const CXXTemporary *temporary,
                                      QualType tempType, Address ptr) {
  pushDestroy(NormalAndEHCleanup, ptr, tempType, destroyCXXObject);
}

//===----------------------------------------------------------------------===//
// EHScopeStack
//===----------------------------------------------------------------------===//

void EHScopeStack::Cleanup::anchor() {}

EHScopeStack::stable_iterator
EHScopeStack::getInnermostActiveNormalCleanup() const {
  stable_iterator si = getInnermostNormalCleanup();
  stable_iterator se = stable_end();
  while (si != se) {
    EHCleanupScope &cleanup = llvm::cast<EHCleanupScope>(*find(si));
    if (cleanup.isActive())
      return si;
    si = cleanup.getEnclosingNormalCleanup();
  }
  return stable_end();
}

/// Push an entry of the given size onto this protected-scope stack.
char *EHScopeStack::allocate(size_t size) {
  size = llvm::alignTo(size, ScopeStackAlignment);
  if (!startOfBuffer) {
    unsigned capacity = llvm::PowerOf2Ceil(std::max(size, 1024ul));
    startOfBuffer = std::make_unique<char[]>(capacity);
    startOfData = endOfBuffer = startOfBuffer.get() + capacity;
  } else if (static_cast<size_t>(startOfData - startOfBuffer.get()) < size) {
    unsigned currentCapacity = endOfBuffer - startOfBuffer.get();
    unsigned usedCapacity =
        currentCapacity - (startOfData - startOfBuffer.get());
    unsigned requiredCapacity = usedCapacity + size;
    // We know from the 'else if' condition that requiredCapacity is greater
    // than currentCapacity.
    unsigned newCapacity = llvm::PowerOf2Ceil(requiredCapacity);

    std::unique_ptr<char[]> newStartOfBuffer =
        std::make_unique<char[]>(newCapacity);
    char *newEndOfBuffer = newStartOfBuffer.get() + newCapacity;
    char *newStartOfData = newEndOfBuffer - usedCapacity;
    memcpy(newStartOfData, startOfData, usedCapacity);
    startOfBuffer.swap(newStartOfBuffer);
    endOfBuffer = newEndOfBuffer;
    startOfData = newStartOfData;
  }

  assert(startOfBuffer.get() + size <= startOfData);
  startOfData -= size;
  return startOfData;
}

void EHScopeStack::deallocate(size_t size) {
  startOfData += llvm::alignTo(size, ScopeStackAlignment);
}

/// Remove any 'null' fixups on the stack.  However, we can't pop more
/// fixups than the fixup depth on the innermost normal cleanup, or
/// else fixups that we try to add to that cleanup will end up in the
/// wrong place.  We *could* try to shrink fixup depths, but that's
/// actually a lot of work for little benefit.
void EHScopeStack::popNullFixups() {
  // We expect this to only be called when there's still an innermost
  // normal cleanup;  otherwise there really shouldn't be any fixups.
  cgf->cgm.errorNYI("popNullFixups");
}

void *EHScopeStack::pushCleanup(CleanupKind kind, size_t size) {
  char *buffer = allocate(EHCleanupScope::getSizeForCleanupSize(size));
  bool isNormalCleanup = kind & NormalCleanup;
  bool isEHCleanup = kind & EHCleanup;
  bool isLifetimeMarker = kind & LifetimeMarker;

  assert(!cir::MissingFeatures::innermostEHScope());

  EHCleanupScope *scope = new (buffer) EHCleanupScope(
      size, branchFixups.size(), innermostNormalCleanup, innermostEHScope);

  if (isNormalCleanup)
    innermostNormalCleanup = stable_begin();

  if (isLifetimeMarker)
    cgf->cgm.errorNYI("push lifetime marker cleanup");

  // With Windows -EHa, Invoke llvm.seh.scope.begin() for EHCleanup
  if (cgf->getLangOpts().EHAsynch && isEHCleanup && !isLifetimeMarker &&
      cgf->getTarget().getCXXABI().isMicrosoft())
    cgf->cgm.errorNYI("push seh cleanup");

  return scope->getCleanupBuffer();
}

void EHScopeStack::popCleanup() {
  assert(!empty() && "popping exception stack when not empty");

  assert(isa<EHCleanupScope>(*begin()));
  EHCleanupScope &cleanup = cast<EHCleanupScope>(*begin());
  innermostNormalCleanup = cleanup.getEnclosingNormalCleanup();
  deallocate(cleanup.getAllocatedSize());

  // Destroy the cleanup.
  cleanup.destroy();

  // Check whether we can shrink the branch-fixups stack.
  if (!branchFixups.empty()) {
    // If we no longer have any normal cleanups, all the fixups are
    // complete.
    if (!hasNormalCleanups()) {
      branchFixups.clear();
    } else {
      // Otherwise we can still trim out unnecessary nulls.
      popNullFixups();
    }
  }
}

bool EHScopeStack::requiresCatchOrCleanup() const {
  for (stable_iterator si = getInnermostEHScope(); si != stable_end();) {
    if (auto *cleanup = dyn_cast<EHCleanupScope>(&*find(si))) {
      if (cleanup->isLifetimeMarker()) {
        // Skip lifetime markers and continue from the enclosing EH scope
        assert(!cir::MissingFeatures::emitLifetimeMarkers());
        continue;
      }
    }
    return true;
  }
  return false;
}

EHCatchScope *EHScopeStack::pushCatch(unsigned numHandlers) {
  char *buffer = allocate(EHCatchScope::getSizeForNumHandlers(numHandlers));
  EHCatchScope *scope =
      new (buffer) EHCatchScope(numHandlers, innermostEHScope);
  innermostEHScope = stable_begin();
  return scope;
}

static void emitCleanup(CIRGenFunction &cgf, EHScopeStack::Cleanup *cleanup) {
  // Ask the cleanup to emit itself.
  assert(cgf.haveInsertPoint() && "expected insertion point");
  assert(!cir::MissingFeatures::ehCleanupFlags());
  cleanup->emit(cgf);
  assert(cgf.haveInsertPoint() && "cleanup ended with no insertion point?");
}

static mlir::Block *createNormalEntry(CIRGenFunction &cgf,
                                      EHCleanupScope &scope) {
  assert(scope.isNormalCleanup());
  mlir::Block *entry = scope.getNormalBlock();
  if (!entry) {
    mlir::OpBuilder::InsertionGuard guard(cgf.getBuilder());
    entry = cgf.curLexScope->getOrCreateCleanupBlock(cgf.getBuilder());
    scope.setNormalBlock(entry);
  }
  return entry;
}

/// Pops a cleanup block. If the block includes a normal cleanup, the
/// current insertion point is threaded through the cleanup, as are
/// any branch fixups on the cleanup.
void CIRGenFunction::popCleanupBlock() {
  assert(!ehStack.empty() && "cleanup stack is empty!");
  assert(isa<EHCleanupScope>(*ehStack.begin()) && "top not a cleanup!");
  EHCleanupScope &scope = cast<EHCleanupScope>(*ehStack.begin());
  assert(scope.getFixupDepth() <= ehStack.getNumBranchFixups());

  // Remember activation information.
  bool isActive = scope.isActive();

  // - whether there are branch fix-ups through this cleanup
  unsigned fixupDepth = scope.getFixupDepth();
  bool hasFixups = ehStack.getNumBranchFixups() != fixupDepth;

  // - whether there's a fallthrough
  mlir::Block *fallthroughSource = builder.getInsertionBlock();
  bool hasFallthrough = fallthroughSource != nullptr && isActive;

  bool requiresNormalCleanup =
      scope.isNormalCleanup() && (hasFixups || hasFallthrough);

  // If we don't need the cleanup at all, we're done.
  assert(!cir::MissingFeatures::ehCleanupScopeRequiresEHCleanup());
  if (!requiresNormalCleanup) {
    ehStack.popCleanup();
    return;
  }

  // Copy the cleanup emission data out.  This uses either a stack
  // array or malloc'd memory, depending on the size, which is
  // behavior that SmallVector would provide, if we could use it
  // here. Unfortunately, if you ask for a SmallVector<char>, the
  // alignment isn't sufficient.
  auto *cleanupSource = reinterpret_cast<char *>(scope.getCleanupBuffer());
  alignas(EHScopeStack::ScopeStackAlignment) char
      cleanupBufferStack[8 * sizeof(void *)];
  std::unique_ptr<char[]> cleanupBufferHeap;
  size_t cleanupSize = scope.getCleanupSize();
  EHScopeStack::Cleanup *cleanup;

  // This is necessary because we are going to deallocate the cleanup
  // (in popCleanup) before we emit it.
  if (cleanupSize <= sizeof(cleanupBufferStack)) {
    memcpy(cleanupBufferStack, cleanupSource, cleanupSize);
    cleanup = reinterpret_cast<EHScopeStack::Cleanup *>(cleanupBufferStack);
  } else {
    cleanupBufferHeap.reset(new char[cleanupSize]);
    memcpy(cleanupBufferHeap.get(), cleanupSource, cleanupSize);
    cleanup =
        reinterpret_cast<EHScopeStack::Cleanup *>(cleanupBufferHeap.get());
  }

  assert(!cir::MissingFeatures::ehCleanupFlags());

  // If we have a fallthrough and no other need for the cleanup,
  // emit it directly.
  if (hasFallthrough && !hasFixups) {
    assert(!cir::MissingFeatures::ehCleanupScopeRequiresEHCleanup());
    ehStack.popCleanup();
    scope.markEmitted();
    emitCleanup(*this, cleanup);
  } else {
    // Otherwise, the best approach is to thread everything through
    // the cleanup block and then try to clean up after ourselves.

    // Force the entry block to exist.
    mlir::Block *normalEntry = createNormalEntry(*this, scope);

    // I.  Set up the fallthrough edge in.
    mlir::OpBuilder::InsertPoint savedInactiveFallthroughIP;

    // If there's a fallthrough, we need to store the cleanup
    // destination index. For fall-throughs this is always zero.
    if (hasFallthrough) {
      assert(!cir::MissingFeatures::ehCleanupHasPrebranchedFallthrough());

    } else if (fallthroughSource) {
      // Otherwise, save and clear the IP if we don't have fallthrough
      // because the cleanup is inactive.
      assert(!isActive && "source without fallthrough for active cleanup");
      savedInactiveFallthroughIP = builder.saveInsertionPoint();
    }

    // II.  Emit the entry block.  This implicitly branches to it if
    // we have fallthrough.  All the fixups and existing branches
    // should already be branched to it.
    builder.setInsertionPointToEnd(normalEntry);

    // intercept normal cleanup to mark SEH scope end
    assert(!cir::MissingFeatures::ehCleanupScopeRequiresEHCleanup());

    // III.  Figure out where we're going and build the cleanup
    // epilogue.
    bool hasEnclosingCleanups =
        (scope.getEnclosingNormalCleanup() != ehStack.stable_end());

    // Compute the branch-through dest if we need it:
    //   - if there are branch-throughs threaded through the scope
    //   - if fall-through is a branch-through
    //   - if there are fixups that will be optimistically forwarded
    //     to the enclosing cleanup
    assert(!cir::MissingFeatures::cleanupBranchThrough());
    if (hasFixups && hasEnclosingCleanups)
      cgm.errorNYI("cleanup branch-through dest");

    mlir::Block *fallthroughDest = nullptr;

    // If there's exactly one branch-after and no other threads,
    // we can route it without a switch.
    // Skip for SEH, since ExitSwitch is used to generate code to indicate
    // abnormal termination. (SEH: Except _leave and fall-through at
    // the end, all other exits in a _try (return/goto/continue/break)
    // are considered as abnormal terminations, using NormalCleanupDestSlot
    // to indicate abnormal termination)
    assert(!cir::MissingFeatures::cleanupBranchThrough());
    assert(!cir::MissingFeatures::ehCleanupScopeRequiresEHCleanup());

    // IV.  Pop the cleanup and emit it.
    scope.markEmitted();
    ehStack.popCleanup();
    assert(ehStack.hasNormalCleanups() == hasEnclosingCleanups);

    emitCleanup(*this, cleanup);

    // Append the prepared cleanup prologue from above.
    assert(!cir::MissingFeatures::cleanupAppendInsts());

    // Optimistically hope that any fixups will continue falling through.
    if (fixupDepth != ehStack.getNumBranchFixups())
      cgm.errorNYI("cleanup fixup depth mismatch");

    // V.  Set up the fallthrough edge out.

    // Case 1: a fallthrough source exists but doesn't branch to the
    // cleanup because the cleanup is inactive.
    if (!hasFallthrough && fallthroughSource) {
      // Prebranched fallthrough was forwarded earlier.
      // Non-prebranched fallthrough doesn't need to be forwarded.
      // Either way, all we need to do is restore the IP we cleared before.
      assert(!isActive);
      cgm.errorNYI("cleanup inactive fallthrough");

      // Case 2: a fallthrough source exists and should branch to the
      // cleanup, but we're not supposed to branch through to the next
      // cleanup.
    } else if (hasFallthrough && fallthroughDest) {
      cgm.errorNYI("cleanup fallthrough destination");

      // Case 3: a fallthrough source exists and should branch to the
      // cleanup and then through to the next.
    } else if (hasFallthrough) {
      // Everything is already set up for this.

      // Case 4: no fallthrough source exists.
    } else {
      // FIXME(cir): should we clear insertion point here?
    }

    // VI.  Assorted cleaning.

    // Check whether we can merge NormalEntry into a single predecessor.
    // This might invalidate (non-IR) pointers to NormalEntry.
    //
    // If it did invalidate those pointers, and normalEntry was the same
    // as NormalExit, go back and patch up the fixups.
    assert(!cir::MissingFeatures::simplifyCleanupEntry());
  }
}

/// Pops cleanup blocks until the given savepoint is reached.
void CIRGenFunction::popCleanupBlocks(
    EHScopeStack::stable_iterator oldCleanupStackDepth) {
  assert(!cir::MissingFeatures::ehstackBranches());

  // Pop cleanup blocks until we reach the base stack depth for the
  // current scope.
  while (ehStack.stable_begin() != oldCleanupStackDepth) {
    popCleanupBlock();
  }
}
