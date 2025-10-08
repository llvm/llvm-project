//===- LifetimeSafety.h - C++ Lifetime Safety Analysis -*----------- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the entry point for a dataflow-based static analysis
// that checks for C++ lifetime violations.
//
// The analysis is based on the concepts of "origins" and "loans" to track
// pointer lifetimes and detect issues like use-after-free and dangling
// pointers. See the RFC for more details:
// https://discourse.llvm.org/t/rfc-intra-procedural-lifetime-analysis-in-clang/86291
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_CLANG_ANALYSIS_ANALYSES_LIFETIMESAFETY_H
#define LLVM_CLANG_ANALYSIS_ANALYSES_LIFETIMESAFETY_H

#include "clang/Analysis/Analyses/LifetimeSafety/Facts.h"
#include "clang/Analysis/Analyses/LifetimeSafety/LiveOrigins.h"
#include "clang/Analysis/Analyses/LifetimeSafety/LoanPropagation.h"
#include "clang/Analysis/Analyses/LifetimeSafety/Reporter.h"
#include "clang/Analysis/AnalysisDeclContext.h"
#include "clang/Analysis/CFG.h"
#include "llvm/ADT/StringMap.h"
#include <memory>

namespace clang::lifetimes {

/// The main entry point for the analysis.
void runLifetimeSafetyAnalysis(AnalysisDeclContext &AC,
                               LifetimeSafetyReporter *Reporter);

namespace internal {
// Forward declarations of internal types.
struct LifetimeFactory;

/// Running the lifetime safety analysis and querying its results. It
/// encapsulates the various dataflow analyses.
class LifetimeSafetyAnalysis {
public:
  LifetimeSafetyAnalysis(AnalysisDeclContext &AC,
                         LifetimeSafetyReporter *Reporter);
  ~LifetimeSafetyAnalysis();

  void run();

  /// Returns the set of loans an origin holds at a specific program point.
  LoanSet getLoansAtPoint(OriginID OID, ProgramPoint PP) const;

  /// Returns the set of origins that are live at a specific program point,
  /// along with the confidence level of their liveness.
  ///
  /// An origin is considered live if there are potential future uses of that
  /// origin after the given program point. The confidence level indicates
  /// whether the origin is definitely live (Definite) due to being domintated
  /// by a set of uses or only possibly live (Maybe) only on some but not all
  /// control flow paths.
  std::vector<std::pair<OriginID, LivenessKind>>
  getLiveOriginsAtPoint(ProgramPoint PP) const;

  /// Finds the OriginID for a given declaration.
  /// Returns a null optional if not found.
  std::optional<OriginID> getOriginIDForDecl(const ValueDecl *D) const;

  /// Finds the LoanID's for the loan created with the specific variable as
  /// their Path.
  std::vector<LoanID> getLoanIDForVar(const VarDecl *VD) const;

  /// Retrieves program points that were specially marked in the source code
  /// for testing.
  ///
  /// The analysis recognizes special function calls of the form
  /// `void("__lifetime_test_point_<name>")` as test points. This method returns
  /// a map from the annotation string (<name>) to the corresponding
  /// `ProgramPoint`. This allows test harnesses to query the analysis state at
  /// user-defined locations in the code.
  /// \note This is intended for testing only.
  llvm::StringMap<ProgramPoint> getTestPoints() const;

private:
  AnalysisDeclContext &AC;
  LifetimeSafetyReporter *Reporter;
  std::unique_ptr<LifetimeFactory> Factory;
  std::unique_ptr<FactManager> FactMgr;
  std::unique_ptr<LoanPropagationAnalysis> LoanPropagation;
  std::unique_ptr<LiveOriginAnalysis> LiveOrigins;
};
} // namespace internal
} // namespace clang::lifetimes

#endif // LLVM_CLANG_ANALYSIS_ANALYSES_LIFETIMESAFETY_H
