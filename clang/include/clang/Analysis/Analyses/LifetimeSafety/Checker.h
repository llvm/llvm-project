//===- Checker.h - C++ Lifetime Safety Analysis -*----------- C++-*-=========//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file provides the entry point for lifetime checking, which detects
// use-after-free errors by checking if live origins hold loans that have
// expired.
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_CLANG_ANALYSIS_ANALYSES_LIFETIMESAFETY_CHECKER_H
#define LLVM_CLANG_ANALYSIS_ANALYSES_LIFETIMESAFETY_CHECKER_H

#include "clang/Analysis/Analyses/LifetimeSafety/Facts.h"
#include "clang/Analysis/Analyses/LifetimeSafety/LiveOrigins.h"
#include "clang/Analysis/Analyses/LifetimeSafety/LoanPropagation.h"
#include "clang/Analysis/Analyses/LifetimeSafety/Reporter.h"

namespace clang {
namespace lifetimes {
namespace internal {

/// Runs the lifetime checker, which detects use-after-free errors by
/// examining loan expiration points and checking if any live origins hold
/// the expired loan.
void runLifetimeChecker(LoanPropagationAnalysis &LoanPropagation,
                        LiveOriginAnalysis &LiveOrigins, FactManager &FactMgr,
                        AnalysisDeclContext &ADC,
                        LifetimeSafetyReporter *Reporter);

} // namespace internal
} // namespace lifetimes
} // namespace clang

#endif // LLVM_CLANG_ANALYSIS_ANALYSES_LIFETIMESAFETY_CHECKER_H
