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
  assert(!ehStack.cleanupStack.empty() && "cleanup stack is empty!");
  mlir::OpBuilder::InsertionGuard guard(builder);
  std::unique_ptr<EHScopeStack::Cleanup> cleanup =
      ehStack.cleanupStack.pop_back_val();

  assert(!cir::MissingFeatures::ehCleanupFlags());
  mlir::Block *cleanupEntry = getCurCleanupBlock(*this);
  builder.setInsertionPointToEnd(cleanupEntry);
  cleanup->emit(*this);
}

/// Pops cleanup blocks until the given savepoint is reached.
void CIRGenFunction::popCleanupBlocks(size_t oldCleanupStackDepth) {
  assert(!cir::MissingFeatures::ehstackBranches());

  assert(ehStack.getStackDepth() >= oldCleanupStackDepth);

  // Pop cleanup blocks until we reach the base stack depth for the
  // current scope.
  while (ehStack.getStackDepth() > oldCleanupStackDepth) {
    popCleanupBlock();
  }
}
