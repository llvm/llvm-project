//===- DataflowLattice.h ----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file defines base types for building lattices to be used in dataflow
//  analyses that run over Control-Flow Graphs (CFGs).
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_ANALYSIS_FLOWSENSITIVE_DATAFLOWLATTICE_H
#define LLVM_CLANG_ANALYSIS_FLOWSENSITIVE_DATAFLOWLATTICE_H

#include "llvm/Support/Casting.h"
#include "llvm/Support/ExtensibleRTTI.h"
#include <memory>

namespace clang {
namespace dataflow {

/// Effect indicating whether a lattice operation resulted in a new value.
enum class LatticeEffect {
  Unchanged,
  Changed,
};
// DEPRECATED. Use `LatticeEffect`.
using LatticeJoinEffect = LatticeEffect;

class DataflowLattice
    : public llvm::RTTIExtends<DataflowLattice, llvm::RTTIRoot> {
public:
  /// For RTTI.
  inline static char ID = 0;

  DataflowLattice() = default;

  /// Joins the object with `Other` by computing their least upper bound,
  /// modifies the object if necessary, and returns an effect indicating whether
  /// any changes were made to it.
  virtual LatticeEffect join(const DataflowLattice &Other) = 0;

  virtual std::unique_ptr<DataflowLattice> clone() = 0;

  /// Replaces this lattice element with one that approximates it, given the
  /// previous element at the current program point.  The approximation should
  /// be chosen so that the analysis can reach a fixed point more quickly than
  /// iterated application of the transfer function alone. The previous value is
  /// provided to inform the choice of widened value. The function must also
  /// serve as a comparison operation, by indicating whether the widened value
  /// is equivalent to the previous value with the returned `LatticeJoinEffect`.
  ///
  /// Overriding `widen` is optional -- it is only needed to either accelerate
  /// convergence (for lattices with non-trivial height) or guarantee
  /// convergence (for lattices with infinite height). The default
  /// implementation simply checks for equality.
  ///
  /// Returns an indication of whether any changes were made to the object. This
  /// saves a separate call to `isEqual` after the widening.
  virtual LatticeEffect widen(const DataflowLattice &Previous) {
    return isEqual(Previous) ? LatticeEffect::Unchanged
                             : LatticeEffect::Changed;
  }

  /// Returns true if and only if the two given type-erased lattice elements are
  /// equal.
  virtual bool isEqual(const DataflowLattice &) const = 0;
};

using DataflowLatticePtr = std::unique_ptr<DataflowLattice>;

} // namespace dataflow
} // namespace clang

#endif // LLVM_CLANG_ANALYSIS_FLOWSENSITIVE_DATAFLOWLATTICE_H
