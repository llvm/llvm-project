//===- LoopNestPattern.h - Loop nest classification -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Classification of a counted two-loop nest into the iteration domain shapes
// the CIR loop optimizer knows how to reason about.
//
// Classification is syntactic. It reports the domain implied by the loop
// control. Early exits may reduce the executed iteration set. Everything
// else belongs to legality, including perfect nesting, reachability and the
// memory the body touches.
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_LIB_CIR_DIALECT_TRANSFORMS_LOOPOPT_LOOPNESTPATTERN_H
#define CLANG_LIB_CIR_DIALECT_TRANSFORMS_LOOPOPT_LOOPNESTPATTERN_H

#include "LoopAnalysis.h"

#include "mlir/IR/Dominance.h"

#include <optional>

namespace cir {
namespace loopopt {

/// How the inner loop's iteration space relates to the outer loop.
enum class LoopNestPatternKind {
  /// The inner limit is an affine function of the outer induction variable,
  /// for example `j < i`, `j < i + k` or `j < 2 * i`.
  InnerAffineUpper,

  /// The inner loop starts at the outer induction variable and both loops
  /// share an upper limit, for example `for (j = i; j < n; ++j)`.
  OuterIVInnerStart,

  /// The inner limit compares the product of both induction variables
  /// against a constant, for example `j * i < n`.
  InnerProductUpper,

  /// The inner loop starts at a value that is fixed for the whole nest, so
  /// the iteration space is rectangular but not constant.
  InvariantInnerStart
};

/// The inner limit written as `coefficient * outerIV + offset`.
struct AffineOuterIVRelation {
  llvm::APSInt coefficient;
  llvm::APSInt offset;
};

/// For InnerAffineUpper, `affine` stores the normalized relation. It is
/// empty for all other kinds.
struct LoopNestPattern {
  CountedLoop outer;
  CountedLoop inner;
  LoopNestPatternKind kind;

  std::optional<AffineOuterIVRelation> affine;
};

/// Classify a counted pair, failing when it matches no known shape. inner
/// must be the loop getSingleInnerFor returns for outer. The dominance info
/// must cover both loops and any definition the matcher inspects, such as
/// the store behind an invariant start.
mlir::FailureOr<LoopNestPattern>
matchLoopNestPattern(const CountedLoop &outer, const CountedLoop &inner,
                     mlir::DominanceInfo &dominance);

} // namespace loopopt
} // namespace cir

#endif // CLANG_LIB_CIR_DIALECT_TRANSFORMS_LOOPOPT_LOOPNESTPATTERN_H
