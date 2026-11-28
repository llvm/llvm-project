//===-- PointerNullabilityAnalysis.h - Nullability analysis -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_ANALYSIS_FLOWSENSITIVE_MODELS_NULLABILITY_POINTERNULLABILITYANALYSIS_H
#define CLANG_ANALYSIS_FLOWSENSITIVE_MODELS_NULLABILITY_POINTERNULLABILITYANALYSIS_H

#include "clang/AST/ASTContext.h"
#include "clang/AST/Type.h"
#include "clang/Analysis/CFG.h"
#include "clang/Analysis/FlowSensitive/CFGMatchSwitch.h"
#include "clang/Analysis/FlowSensitive/DataflowAnalysis.h"
#include "clang/Analysis/FlowSensitive/DataflowAnalysisContext.h"
#include "clang/Analysis/FlowSensitive/DataflowEnvironment.h"
#include "clang/Analysis/FlowSensitive/MatchSwitch.h"
#include "clang/Analysis/FlowSensitive/NoopLattice.h"
#include "clang/Analysis/FlowSensitive/Value.h"

namespace clang::dataflow::nullability {

// The analysis does not need any lattice state of its own yet. Pointer
// nullability is tracked as properties on `PointerValue`s in the `Environment`.
// We'll use this lattice later to store non-flow-sensitive state and to
// cache const member call results.
using PointerNullabilityLattice = NoopLattice;

/// Analyzes constructs in the source code to collect nullability information
/// about pointers at each program point. This analysis and the corresponding
/// lattice were based on the gradual analysis in 'Estep, Sam, Jenna Wise,
/// Jonathan Aldrich, Éric Tanter, Johannes Bader, and Joshua Sunshine. "Gradual
/// Program Analysis for Null Pointers." In 35th European Conference on
/// Object-Oriented Programming (ECOOP). 2021.'
class PointerNullabilityAnalysis final
    : public DataflowAnalysis<PointerNullabilityAnalysis,
                              PointerNullabilityLattice> {
public:
  explicit PointerNullabilityAnalysis(ASTContext &Context, Environment &Env);

  PointerNullabilityLattice initialElement() { return {}; }

  void transfer(const CFGElement &Elt, PointerNullabilityLattice &Lattice,
                Environment &Env);

  void join(QualType Type, const Value &Val1, const Environment &Env1,
            const Value &Val2, const Environment &Env2, Value &MergedVal,
            Environment &MergedEnv) override;

  // TODO: Add `compare` and `widen` overrides, as in
  // https://github.com/google/crubit/blob/5f9ae7da1aa70e5a1c0e536baf32e5ec56a7c003/nullability/pointer_nullability_analysis.cc#L239
  // Until then, this only handles loop-free code precisely: the framework's
  // default comparison treats any two pointer values at a loop head as
  // equivalent, so the loop body is not re-analyzed when a pointer's
  // nullability changes along the back edge.

private:
  // Transfers (flow-sensitive) value properties through statements.
  CFGMatchSwitch<TransferState<PointerNullabilityLattice>> ValueTransferer;
};

} // namespace clang::dataflow::nullability

#endif // CLANG_ANALYSIS_FLOWSENSITIVE_MODELS_NULLABILITY_POINTERNULLABILITYANALYSIS_H
