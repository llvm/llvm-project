//===- Solver.h -------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file defines an interface for a SAT solver that can be used by
//  dataflow analyses.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_ANALYSIS_FLOWSENSITIVE_SOLVER_H
#define LLVM_CLANG_ANALYSIS_FLOWSENSITIVE_SOLVER_H

#include "clang/Analysis/FlowSensitive/Value.h"
#include "llvm/ADT/DenseSet.h"

namespace clang {
namespace dataflow {

/// An interface for a SAT solver that can be used by dataflow analyses.
class Solver {
public:
  enum class Result {
    /// Indicates that there exists a satisfying assignment for a boolean
    /// formula.
    Satisfiable,

    /// Indicates that there is no satisfying assignment for a boolean formula.
    Unsatisfiable,

    /// Indicates that the solver gave up trying to find a satisfying assignment
    /// for a boolean formula.
    TimedOut,
  };

  virtual ~Solver() = default;

  /// Checks if the conjunction of `Vals` is satisfiable and returns the
  /// corresponding result.
  ///
  /// Requirements:
  ///
  ///  All elements in `Vals` must not be null.
  ///
  /// FIXME: Consider returning a model in case the conjunction of `Vals` is
  /// satisfiable so that it can be used to generate warning messages.
  virtual Result solve(llvm::DenseSet<BoolValue *> Vals) = 0;
};

} // namespace dataflow
} // namespace clang

#endif // LLVM_CLANG_ANALYSIS_FLOWSENSITIVE_SOLVER_H
