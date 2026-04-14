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

#include "clang/AST/Decl.h"
#include "clang/Analysis/Analyses/LifetimeSafety/Facts.h"
#include "clang/Analysis/Analyses/LifetimeSafety/LiveOrigins.h"
#include "clang/Analysis/Analyses/LifetimeSafety/LoanPropagation.h"
#include "clang/Analysis/Analyses/LifetimeSafety/MovedLoans.h"
#include "clang/Analysis/Analyses/LifetimeSafety/Origins.h"
#include "clang/Analysis/AnalysisDeclContext.h"

namespace clang::lifetimes {

using AssignmentPair = std::pair<OriginDestExpr, const Expr *>;

struct ExprPrintingResult {
  llvm::StringRef Str;
  const Expr *CurrExpr;
};

void formatLoanEntityForSema(LoanEntity IssueEntity,
                             llvm::SmallVectorImpl<char> &IssueMsg);
void formatSrcExprForSema(
    const Expr *SrcExpr, llvm::SmallVectorImpl<ExprPrintingResult> &SrcMsgList);
} // namespace clang::lifetimes

namespace clang::lifetimes::internal {

struct AliasAssignmentSearchResult {
  const bool SearchComplete;
  const OriginDestExpr LastDestDeclOrExpr;
  const std::optional<OriginID> IssueOriginID;
};

struct AssignmentQueryContext {
  const LoanPropagationAnalysis &LoanPropagation;
  const MovedLoansAnalysis &MovedLoans;
  const LiveOriginsAnalysis &LiveOrigins;
  FactManager &FactMgr;
  AnalysisDeclContext &ADC;
};

/// Get assignment history when an error is detected.
///
/// To help users understand the data flow, we track where the problematic
/// address originated.
///
/// `StartBlock` denotes the beginning block of the search, while `IssueExpr`
/// represents the ending block.
/// Note that `IssueExpr` can be null, as certain errors may not provide a
/// specific expression. The search termination condition is independent of
/// `IssueExpr`; it serves only as a hint to assist specific search strategies.
void getAliasList(const AssignmentQueryContext &Context,
                  llvm::SmallVectorImpl<AssignmentPair> &AssignmentList,
                  const Fact *CausingFact, const LoanID End,
                  const CFGBlock *StartBlock, const Expr *IssueExpr);
} // namespace clang::lifetimes::internal

#endif
