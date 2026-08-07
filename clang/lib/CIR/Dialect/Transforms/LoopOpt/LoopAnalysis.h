//===- LoopAnalysis.h - Counted loop recognition ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Recognizes canonical CIR loop-control shapes. The analysis identifies the
// induction slot, initialization, condition, and unit step. The induction
// slot must not escape and may be written inside the loop only by the
// matched step.
//
// Body control flow and unrelated effects are not analyzed. The reported
// domain is implied by the loop control. Legality is outside this analysis.
// The trip count is not proven fixed here, because the non-induction side of
// the exit test may still move. The classifiers in LoopNestPattern.h settle
// that.
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_LIB_CIR_DIALECT_TRANSFORMS_LOOPOPT_LOOPANALYSIS_H
#define CLANG_LIB_CIR_DIALECT_TRANSFORMS_LOOPOPT_LOOPANALYSIS_H

#include "mlir/Support/LogicalResult.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "llvm/ADT/APSInt.h"

namespace cir {
namespace loopopt {

enum class StepDirection { Increment, Decrement };

/// The exit test split around the induction variable. ivDependentExpr is the
/// whole side that reads it, such as `j` or `j * i`. nonIvExpr may read an
/// enclosing loop's variable but not this one's. The predicate reads as
/// though ivDependentExpr were on the left, so `n > j` is recorded as
/// `j < n`.
struct ControlComparison {
  mlir::Value ivDependentExpr;
  mlir::Value nonIvExpr;
  cir::CmpOpKind predicate;
};

/// A loop that runs its exit compare every iteration and advances one slot
/// by exactly one unit step, with no other write to that slot inside the
/// loop.
struct CountedLoop {
  cir::ForOp forOp;
  mlir::Value ivSlot;

  mlir::Value initialExpr;

  ControlComparison condition;

  StepDirection direction;
};

/// True for a load or store that is neither volatile nor atomic.
bool isOrdinaryAccess(cir::LoadOp load);
bool isOrdinaryAccess(cir::StoreOp store);

/// True for an add or subtract that clamps instead of wrapping.
bool isSaturating(mlir::Operation *op);

/// True for the operations a loop control expression may be built from.
/// These are an integer constant, an ordinary load, non-saturating add,
/// subtract and multiply, and a divide that folds to a constant. Casts are
/// not supported.
bool isSupportedControlOp(mlir::Operation *op);

/// Fold an integer expression built from constants, preserving width and
/// signedness. Overflow, division by zero and any unsupported operation fail
/// rather than wrap or guess.
mlir::FailureOr<llvm::APSInt> evaluateConstantIntExpr(mlir::Value value);

/// Integer arithmetic that fails on overflow or on mismatched operands.
mlir::FailureOr<llvm::APSInt> checkedAdd(const llvm::APSInt &lhs,
                                         const llvm::APSInt &rhs);
mlir::FailureOr<llvm::APSInt> checkedSub(const llvm::APSInt &lhs,
                                         const llvm::APSInt &rhs);
mlir::FailureOr<llvm::APSInt> checkedMul(const llvm::APSInt &lhs,
                                         const llvm::APSInt &rhs);

bool isOrdinaryLoadOfSlotIn(mlir::Value value, mlir::Value slot,
                            mlir::Operation *scope);

mlir::FailureOr<CountedLoop> matchCountedLoop(cir::ForOp forOp);

/// The sole cir.for whose nearest enclosing loop is forOp. Looks through
/// scopes and ignores loops nested under another loop. Null when the body
/// holds no such loop, or more than one.
cir::ForOp getSingleInnerFor(cir::ForOp forOp);

} // namespace loopopt
} // namespace cir

#endif // CLANG_LIB_CIR_DIALECT_TRANSFORMS_LOOPOPT_LOOPANALYSIS_H
