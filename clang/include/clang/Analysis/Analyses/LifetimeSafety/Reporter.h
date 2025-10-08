//===- Reporter.h - Lifetime Safety Error Reporter -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the LifetimeSafetyReporter interface for reporting
// lifetime safety violations and the Confidence enum for diagnostic severity.
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_CLANG_ANALYSIS_ANALYSES_LIFETIMESAFETY_REPORTER_H
#define LLVM_CLANG_ANALYSIS_ANALYSES_LIFETIMESAFETY_REPORTER_H

#include "clang/AST/Expr.h"
#include "clang/Basic/SourceLocation.h"

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
} // namespace clang::lifetimes
#endif // LLVM_CLANG_ANALYSIS_ANALYSES_LIFETIMESAFETY_REPORTER_H
