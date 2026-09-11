//===- TieredSolver.h -------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file defines a SAT solver implementation that can be used by dataflow
//  analyses.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_ANALYSIS_FLOWSENSITIVE_TIEREDSOLVER_H
#define LLVM_CLANG_ANALYSIS_FLOWSENSITIVE_TIEREDSOLVER_H

#include "clang/Analysis/FlowSensitive/Formula.h"
#include "clang/Analysis/FlowSensitive/Solver.h"
#include "llvm/ADT/ArrayRef.h"

namespace clang {
namespace dataflow {

/// A `Solver` implementation that delegates to a sequence of other solvers.
///
/// The first solver that does not time out is used to produce the result.
///
/// The intent is that we can combine a solver that solves simple problems
/// quickly but times out on harder problems (e.g. `WatchedLiteralsSolver`) with
/// a solver that solves simple problems more slowly but can solve harder
/// problems without timing out.
class TieredSolver : public Solver {
public:
  explicit TieredSolver(llvm::SmallVector<std::unique_ptr<Solver>> Tiers)
      : Tiers_(std::move(Tiers)) {
    assert(!Tiers_.empty());
  }

  Result solve(llvm::ArrayRef<const Formula *> Constraints) override {
    for (const auto &Tier : Tiers_) {
      Result Res = Tier->solve(Constraints);
      if (Res.getStatus() != Result::Status::TimedOut)
        return Res;
    }

    return Result::TimedOut();
  }

  bool reachedLimit() const override {
    for (const auto &Tier : Tiers_)
      if (!Tier->reachedLimit())
        return false;

    return true;
  }

private:
  llvm::SmallVector<std::unique_ptr<Solver>> Tiers_;
};

} // namespace dataflow
} // namespace clang

#endif // LLVM_CLANG_ANALYSIS_FLOWSENSITIVE_TIEREDSOLVER_H
