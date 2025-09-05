//===- ExpressionSimplification.cpp - Simplify HLFIR expressions ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/HLFIR/HLFIROps.h"
#include "flang/Optimizer/HLFIR/Passes.h"
#include "llvm/Support/DebugLog.h"

namespace hlfir {
#define GEN_PASS_DEF_EXPRESSIONSIMPLIFICATION
#include "flang/Optimizer/HLFIR/Passes.h.inc"
} // namespace hlfir

#define DEBUG_TYPE "expression-simplification"

static void removeOp(mlir::Operation *op) {
  op->dropAllReferences();
  op->dropAllUses();
  op->erase();
}

// Get the first user of `op`.
// Note that we consider the first user to be the one on the lowest line of
// the emitted HLFIR. The user iterator considers the opposite.
template <typename UserOp>
static UserOp getFirstUser(mlir::Operation *op) {
  auto it = op->user_begin(), end = op->user_end(), prev = it;
  for (; it != end; prev = it++)
    ;
  if (prev != end)
    if (auto userOp = mlir::dyn_cast<UserOp>(*prev))
      return userOp;
  return {};
}

// Get the last user of `op`.
// Note that we consider the last user to be the one on the highest line of
// the emitted HLFIR. The user iterator considers the opposite.
template <typename UserOp>
static UserOp getLastUser(mlir::Operation *op) {
  if (!op->getUsers().empty())
    if (auto userOp = mlir::dyn_cast<UserOp>(*op->user_begin()))
      return userOp;
  return {};
}

namespace {

// This class analyzes a trimmed character and removes the trim operation if
// its result is not used elsewhere.
class TrimRemover {
public:
  TrimRemover(mlir::Value charVal) : charVal(charVal) {}
  TrimRemover(const TrimRemover &) = delete;

  bool charWasTrimmed();
  void removeTrim();

private:
  mlir::Value charVal;
  hlfir::CharTrimOp trimOp;
  hlfir::CmpCharOp cmpCharOp;
  hlfir::DestroyOp destroyOp;
};

bool TrimRemover::charWasTrimmed() {
  LDBG() << "charWasTrimmed: " << charVal;

  trimOp = mlir::dyn_cast<hlfir::CharTrimOp>(charVal.getDefiningOp());
  if (!trimOp)
    return false;
  int trimUses = std::distance(trimOp->use_begin(), trimOp->use_end());
  cmpCharOp = getFirstUser<hlfir::CmpCharOp>(trimOp);
  destroyOp = getLastUser<hlfir::DestroyOp>(trimOp);
  return cmpCharOp && destroyOp && trimUses == 2;
}

void TrimRemover::removeTrim() {
  LDBG() << "removeTrim: " << trimOp;

  cmpCharOp->replaceUsesOfWith(trimOp.getResult(), trimOp.getChr());
  removeOp(destroyOp);
  removeOp(trimOp);
}

class ExpressionSimplification
    : public hlfir::impl::ExpressionSimplificationBase<
          ExpressionSimplification> {
public:
  using ExpressionSimplificationBase<
      ExpressionSimplification>::ExpressionSimplificationBase;

  void runOnOperation() override;

private:
  // Simplify character comparisons.
  // Because character comparison appends spaces to the shorter character,
  // calls to trim() that are used only in the comparison can be eliminated.
  //
  // Example:
  // `trim(x) == trim(y)`
  // can be simplified to
  // `x == y`
  void simplifyCmpChar(hlfir::CmpCharOp cmpChar);
};

void ExpressionSimplification::simplifyCmpChar(hlfir::CmpCharOp cmpChar) {
  TrimRemover lhsTrimRem(cmpChar.getLchr());
  TrimRemover rhsTrimRem(cmpChar.getRchr());

  if (lhsTrimRem.charWasTrimmed())
    lhsTrimRem.removeTrim();
  if (rhsTrimRem.charWasTrimmed())
    rhsTrimRem.removeTrim();
}

void ExpressionSimplification::runOnOperation() {
  mlir::ModuleOp module = getOperation();
  module.walk([&](hlfir::CmpCharOp cmpChar) { simplifyCmpChar(cmpChar); });
}

} // namespace
