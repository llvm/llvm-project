//===- AssignmentQuery.cpp - C++ Lifetime Safety Checker --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the LifetimeChecker, which detects use-after-free
// errors by checking if live origins hold loans that have expired.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_ANALYSIS_ANALYSES_ASSIGNMENTQUERY_H
#define LLVM_CLANG_ANALYSIS_ANALYSES_ASSIGNMENTQUERY_H

#include "clang/Analysis/Analyses/LifetimeSafety/Facts.h"
#include "clang/Analysis/Analyses/LifetimeSafety/LoanPropagation.h"

namespace clang::lifetimes::internal {
using AssignmentPair = std::pair<DestOriginEntity, SrcOriginEntity>;

struct AssignmentQueryContext {
  const LoanPropagationAnalysis &LoanPropagation;
  FactManager &FactMgr;
};

/// Get assignment history when an error is detected.
///
/// To help user understand the data flow, we track where the problematic
/// address originated.
void trackAssignmentHistory(
    const AssignmentQueryContext &Context,
    llvm::SmallVectorImpl<AssignmentPair> &AssignmentList,
    const CFGBlock *StartBlock, const OriginID StartOID,
    const LoanID EndLoanID);
} // namespace clang::lifetimes::internal

#endif
