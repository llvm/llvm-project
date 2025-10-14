//===- LifetimeSafety.h - C++ Lifetime Safety Analysis -*----------- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the main entry point and orchestrator for the C++ Lifetime
// Safety Analysis. It coordinates the entire analysis pipeline: fact
// generation, loan propagation, live origins analysis, and enforcement of
// lifetime safety policy.
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
#include "clang/Analysis/AnalysisDeclContext.h"

namespace clang::lifetimes {

/// Enum to track the confidence level of a potential error.
enum class Confidence : uint8_t {
  None,
  Maybe,   // Reported as a potential error (-Wlifetime-safety-strict)
  Definite // Reported as a definite error (-Wlifetime-safety-permissive)
};

class LifetimeSafetyReporter {
public:
  LifetimeSafetyReporter() = default;
  virtual ~LifetimeSafetyReporter() = default;

  virtual void reportUseAfterFree(const Expr *IssueExpr, const Expr *UseExpr,
                                  SourceLocation FreeLoc,
                                  Confidence Confidence) {}
};

/// The main entry point for the analysis.
void runLifetimeSafetyAnalysis(AnalysisDeclContext &AC,
                               LifetimeSafetyReporter *Reporter);

namespace internal {
/// An object to hold the factories for immutable collections, ensuring
/// that all created states share the same underlying memory management.
struct LifetimeFactory {
  OriginLoanMap::Factory OriginMapFactory{/*canonicalize=*/false};
  LoanSet::Factory LoanSetFactory{/*canonicalize=*/false};
  LivenessMap::Factory LivenessMapFactory{/*canonicalize=*/false};
};

/// Running the lifetime safety analysis and querying its results. It
/// encapsulates the various dataflow analyses.
class LifetimeSafetyAnalysis {
public:
  LifetimeSafetyAnalysis(AnalysisDeclContext &AC,
                         LifetimeSafetyReporter *Reporter);

  void run();

  /// \note These are provided only for testing purposes.
  LoanPropagationAnalysis &getLoanPropagation() const {
    return *LoanPropagation;
  }
  LiveOriginsAnalysis &getLiveOrigins() const { return *LiveOrigins; }
  FactManager &getFactManager() { return FactMgr; }

private:
  AnalysisDeclContext &AC;
  LifetimeSafetyReporter *Reporter;
  LifetimeFactory Factory;
  FactManager FactMgr;
  std::unique_ptr<LiveOriginsAnalysis> LiveOrigins;
  std::unique_ptr<LoanPropagationAnalysis> LoanPropagation;
};
} // namespace internal
} // namespace clang::lifetimes

#endif // LLVM_CLANG_ANALYSIS_ANALYSES_LIFETIMESAFETY_H
