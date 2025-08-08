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
  char *buffer = allocate(size);

  // When the full implementation is upstreamed, this will allocate
  // extra memory for and construct a wrapper object that is used to
  // manage the cleanup generation.
  assert(!cir::MissingFeatures::ehCleanupScope());

  return buffer;
}

static mlir::Block *getCurCleanupBlock(CIRGenFunction &cgf) {
  mlir::OpBuilder::InsertionGuard guard(cgf.getBuilder());
  mlir::Block *cleanup =
      cgf.curLexScope->getOrCreateCleanupBlock(cgf.getBuilder());
  return cleanup;
}

/// Pops a cleanup block. If the block includes a normal cleanup, the
/// current insertion point is threaded through the cleanup, as are
/// any branch fixups on the cleanup.
void CIRGenFunction::popCleanupBlock() {
  assert(!ehStack.empty() && "cleanup stack is empty!");

  // The memory for the cleanup continues to be owned by the EHScopeStack
  // allocator, so we just destroy the object rather than attempting to
  // free it.
  EHScopeStack::Cleanup &cleanup = *ehStack.begin();

  // The eventual implementation here will use the EHCleanupScope helper class.
  assert(!cir::MissingFeatures::ehCleanupScope());

  mlir::OpBuilder::InsertionGuard guard(builder);

  assert(!cir::MissingFeatures::ehCleanupFlags());
  mlir::Block *cleanupEntry = getCurCleanupBlock(*this);
  builder.setInsertionPointToEnd(cleanupEntry);
  cleanup.emit(*this);

  ehStack.deallocate(cleanup.getSize());
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
