//===- WatchedLiteralsSolver.h ----------------------------------*- C++ -*-===//
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

#ifndef LLVM_CLANG_ANALYSIS_FLOWSENSITIVE_WATCHEDLITERALSSOLVER_H
#define LLVM_CLANG_ANALYSIS_FLOWSENSITIVE_WATCHEDLITERALSSOLVER_H

#include "clang/Analysis/FlowSensitive/Solver.h"
#include "clang/Analysis/FlowSensitive/Value.h"
#include "llvm/ADT/DenseSet.h"

namespace clang {
namespace dataflow {

/// A SAT solver that is an implementation of Algorithm D from Knuth's The Art
/// of Computer Programming Volume 4: Satisfiability, Fascicle 6. It is based on
/// the Davis-Putnam-Logemann-Loveland (DPLL) algorithm, keeps references to a
/// single "watched" literal per clause, and uses a set of "active" variables
/// for unit propagation.
class WatchedLiteralsSolver : public Solver {
public:
  Result solve(llvm::DenseSet<BoolValue *> Vals) override;
};

} // namespace dataflow
} // namespace clang

#endif // LLVM_CLANG_ANALYSIS_FLOWSENSITIVE_WATCHEDLITERALSSOLVER_H
