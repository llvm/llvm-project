//===-- NoopAnalysis.h ------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file defines a NoopAnalysis class that is used by dataflow analysis
//  tests.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_UNITTESTS_ANALYSIS_FLOWSENSITIVE_NOOPANALYSIS_H
#define LLVM_CLANG_UNITTESTS_ANALYSIS_FLOWSENSITIVE_NOOPANALYSIS_H

#include "clang/AST/ASTContext.h"
#include "clang/AST/Stmt.h"
#include "clang/Analysis/FlowSensitive/DataflowAnalysis.h"
#include "clang/Analysis/FlowSensitive/DataflowEnvironment.h"
#include "clang/Analysis/FlowSensitive/DataflowLattice.h"
#include <ostream>

namespace clang {
namespace dataflow {

class NoopLattice {
public:
  bool operator==(const NoopLattice &) const { return true; }

  LatticeJoinEffect join(const NoopLattice &) {
    return LatticeJoinEffect::Unchanged;
  }
};

inline std::ostream &operator<<(std::ostream &OS, const NoopLattice &) {
  return OS << "noop";
}

class NoopAnalysis : public DataflowAnalysis<NoopAnalysis, NoopLattice> {
public:
  /// `ApplyBuiltinTransfer` controls whether to run the built-in transfer
  /// functions that model memory during the analysis. Their results are not
  /// used by `NoopAnalysis`, but tests that need to inspect the environment
  /// should enable them.
  NoopAnalysis(ASTContext &Context, bool ApplyBuiltinTransfer)
      : DataflowAnalysis<NoopAnalysis, NoopLattice>(Context,
                                                    ApplyBuiltinTransfer) {}

  static NoopLattice initialElement() { return {}; }

  void transfer(const Stmt *S, NoopLattice &E, Environment &Env) {}
};

} // namespace dataflow
} // namespace clang

#endif // LLVM_CLANG_UNITTESTS_ANALYSIS_FLOWSENSITIVE_NOOPANALYSIS_H
