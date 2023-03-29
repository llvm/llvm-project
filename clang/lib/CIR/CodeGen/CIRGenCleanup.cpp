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

//===----------------------------------------------------------------------===//
// EHScopeStack
//===----------------------------------------------------------------------===//

void EHScopeStack::Cleanup::anchor() {}

void *EHScopeStack::pushCleanup(CleanupKind Kind, size_t Size) {
  llvm_unreachable("NYI");
  return nullptr;
}