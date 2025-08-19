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

//===----------------------------------------------------------------------===//
// EHScopeStack
//===----------------------------------------------------------------------===//

void EHScopeStack::Cleanup::anchor() {}

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

void *EHScopeStack::pushCleanup(CleanupKind kind, size_t size) {
  char *buffer = allocate(EHCleanupScope::getSizeForCleanupSize(size));
  bool isEHCleanup = kind & EHCleanup;
  bool isLifetimeMarker = kind & LifetimeMarker;

  assert(!cir::MissingFeatures::innermostEHScope());

  EHCleanupScope *scope = new (buffer) EHCleanupScope(size);

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
  deallocate(cleanup.getAllocatedSize());

  // Destroy the cleanup.
  cleanup.destroy();

  assert(!cir::MissingFeatures::ehCleanupBranchFixups());
}

static void emitCleanup(CIRGenFunction &cgf, EHScopeStack::Cleanup *cleanup) {
  // Ask the cleanup to emit itself.
  assert(cgf.haveInsertPoint() && "expected insertion point");
  assert(!cir::MissingFeatures::ehCleanupFlags());
  cleanup->emit(cgf);
  assert(cgf.haveInsertPoint() && "cleanup ended with no insertion point?");
}

/// Pops a cleanup block. If the block includes a normal cleanup, the
/// current insertion point is threaded through the cleanup, as are
/// any branch fixups on the cleanup.
void CIRGenFunction::popCleanupBlock() {
  assert(!ehStack.empty() && "cleanup stack is empty!");
  assert(isa<EHCleanupScope>(*ehStack.begin()) && "top not a cleanup!");
  EHCleanupScope &scope = cast<EHCleanupScope>(*ehStack.begin());

  // Remember activation information.
  bool isActive = scope.isActive();

  assert(!cir::MissingFeatures::ehCleanupBranchFixups());

  // - whether there's a fallthrough
  mlir::Block *fallthroughSource = builder.getInsertionBlock();
  bool hasFallthrough = fallthroughSource != nullptr && isActive;

  bool requiresNormalCleanup = scope.isNormalCleanup() && hasFallthrough;

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

  ehStack.popCleanup();
  scope.markEmitted();
  emitCleanup(*this, cleanup);
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
