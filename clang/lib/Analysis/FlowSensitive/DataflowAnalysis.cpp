//===-- DataflowAnalysis.cpp ------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file defines functions for the DataflowAnalysis entry points.
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/FlowSensitive/DataflowAnalysis.h"
#include "clang/Analysis/FlowSensitive/Solver.h"
#include "clang/Analysis/FlowSensitive/TieredSolver.h"
#include "clang/Analysis/FlowSensitive/WatchedLiteralsSolver.h"
#include "clang/Analysis/FlowSensitive/Z3Solver.h"
#include "llvm/ADT/SmallVector.h"

namespace clang {
namespace dataflow {

std::unique_ptr<Solver> createDefaultSolver() {
#ifdef LLVM_WITH_Z3
  // When Z3 is available, do a tiering strategy:
  // - a simple solver for simple functions
  // - fall back to a production-grade solver for more complex functions.
  llvm::SmallVector<std::unique_ptr<dataflow::Solver>> Tiers;
  constexpr std::int64_t MaxSATIterations = 5'000'000;
  Tiers.push_back(
      std::make_unique<dataflow::WatchedLiteralsSolver>(MaxSATIterations));

  constexpr std::uint32_t Z3Rlimit = 1'000'000'000;
  Tiers.push_back(std::make_unique<dataflow::Z3Solver>(Z3Rlimit));
  return std::make_unique<TieredSolver>(std::move(Tiers));
#else
  /// Default for the maximum number of SAT solver iterations during analysis.
  /// The value was chosen based on the following observations:
  /// - Non-pathological calls to the solver typically require only a few
  /// hundred
  ///   iterations.
  /// - This limit is still low enough to keep runtimes acceptable (on typical
  ///   machines) in cases where we hit the limit.
  constexpr std::int64_t kDefaultMaxSATIterations = 1'000'000'000;
  return std::make_unique<WatchedLiteralsSolver>(kDefaultMaxSATIterations);
#endif
}

} // namespace dataflow
} // namespace clang
