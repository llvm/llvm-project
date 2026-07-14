//===- ArraySectionAnalyzer.h - Analyze array sections --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_OPTIMIZER_ANALYSIS_ARRAYSECTIONANALYZER_H
#define FORTRAN_OPTIMIZER_ANALYSIS_ARRAYSECTIONANALYZER_H

#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/STLFunctionalExtras.h"

namespace mlir {
class Operation;
class Value;
} // namespace mlir

namespace hlfir {
class ElementalOpInterface;
class DesignateOp;
} // namespace hlfir

namespace fir {
class ArraySectionAnalyzer {
public:
  // The result of the analyzis is one of the values below.
  enum class SlicesOverlapKind {
    // Slices overlap is unknown.
    Unknown,
    // Slices are definitely identical.
    DefinitelyIdentical,
    // Slices are definitely disjoint.
    DefinitelyDisjoint,
    // Slices may be either disjoint or identical,
    // i.e. there is definitely no partial overlap.
    EitherIdenticalOrDisjoint
  };

  /// Optional callback used to teach the analyzer that two SSA values, even
  /// though they are not the same SSA value, are known by the caller to
  /// evaluate to the same value at runtime.
  /// The callback may be called with null values, in which case it must
  /// return false.
  using ValueEquivalenceCallback =
      llvm::function_ref<bool(mlir::Value, mlir::Value)>;

  // Analyzes two hlfir.designate results and returns the overlap kind.
  // The callers may use this method when the alias analysis reports
  // an alias of some kind, so that we can run Fortran specific analysis
  // on the array slices to see if they are identical or disjoint.
  // Note that the alias analysis are not able to give such an answer
  // about the references.
  //
  // \p areKnownEquivalent is an optional callback that the analyzer may use
  // to recognize section subscripts that have the same value even when they
  // are not the same SSA value (e.g., subscripts that are loads of the same
  // Fortran variable from two different regions of the same WHERE construct).
  static SlicesOverlapKind
  analyze(mlir::Value ref1, mlir::Value ref2,
          ValueEquivalenceCallback areKnownEquivalent = nullptr);

  static bool isDesignatingArrayInOrder(hlfir::DesignateOp designate,
                                        hlfir::ElementalOpInterface elemental);

private:
  struct SectionDesc {
    // An array section is described by <lb, ub, stride> tuple.
    // If the designator's subscript is not a triple, then
    // the section descriptor is constructed as <lb, nullptr, nullptr>.
    mlir::Value lb, ub, stride;

    SectionDesc(mlir::Value lb, mlir::Value ub, mlir::Value stride);

    // Normalize the section descriptor:
    //   1. If UB is nullptr, then it is set to LB.
    //   2. If LB==UB, then stride does not matter,
    //      so it is reset to nullptr.
    //   3. If STRIDE==1, then it is reset to nullptr.
    void normalize();

    bool operator==(const SectionDesc &other) const;
  };

  // Given an operand_iterator over the indices operands,
  // read the subscript values and return them as SectionDesc
  // updating the iterator. If isTriplet is true,
  // the subscript is a triplet, and the result is <lb, ub, stride>.
  // Otherwise, the subscript is a scalar index, and the result
  // is <index, nullptr, nullptr>.
  static SectionDesc readSectionDesc(mlir::Operation::operand_iterator &it,
                                     bool isTriplet);

  // Return the ordered lower and upper bounds of the section.
  // If stride is known to be non-negative, then the ordered
  // bounds match the <lb, ub> of the descriptor.
  // If stride is known to be negative, then the ordered
  // bounds are <ub, lb> of the descriptor.
  // If stride is unknown, we cannot deduce any order,
  // so the result is <nullptr, nullptr>
  static std::pair<mlir::Value, mlir::Value>
  getOrderedBounds(const SectionDesc &desc);

  // Given two array sections <lb1, ub1, stride1> and
  // <lb2, ub2, stride2>, return true only if the sections
  // are known to be disjoint.
  //
  // For example, for any positive constant C:
  //   X:Y does not overlap with (Y+C):Z
  //   X:Y does not overlap with Z:(X-C)
  static bool areDisjointSections(const SectionDesc &desc1,
                                  const SectionDesc &desc2);

  // Given two array sections <lb1, ub1, stride1> and
  // <lb2, ub2, stride2>, return true only if the sections
  // are known to be identical.
  //
  // For example:
  //   <x, x, stride>
  //   <x, nullptr, nullptr>
  //
  // These sections are identical, from the point of which array
  // elements are being addresses, even though the shape
  // of the array slices might be different.
  //
  // When \p areKnownEquivalent is provided, it is used to compare the
  // individual section components (lower bound, upper bound and stride)
  // when they are not the same SSA value.
  static bool
  areIdenticalSections(const SectionDesc &desc1, const SectionDesc &desc2,
                       ValueEquivalenceCallback areKnownEquivalent = nullptr);

  // Return true, if v1 is known to be less than v2.
  static bool isLess(mlir::Value v1, mlir::Value v2);
};
} // namespace fir

#endif // FORTRAN_OPTIMIZER_ANALYSIS_ARRAYSECTIONANALYZER_H
