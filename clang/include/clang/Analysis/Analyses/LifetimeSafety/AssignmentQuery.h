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

using OriginDestExpr =
    llvm::PointerUnion<const DeclRefExpr *, const ValueDecl *,
                       const MemberExpr *>;
using LoanEntity = llvm::PointerUnion<const Expr *, const ParmVarDecl *,
                                      const CXXMethodDecl *>;

using AssignmentPair = std::pair<OriginDestExpr, const Expr *>;

struct ExprPrintingResult {
  llvm::StringRef Str;
  const Expr *CurrExpr;
};

void FormatLoanEntityForSema(LoanEntity IssueEntity,
                             llvm::SmallVectorImpl<char> &IssueMsg);
void FormatSrcExprForSema(
    const Expr *SrcExpr, llvm::SmallVectorImpl<ExprPrintingResult> &SrcMsgList);
} // namespace clang::lifetimes

namespace clang::lifetimes::internal {

struct AliasAssignmentSearchResult {
  bool SearchComplete;
  const std::optional<OriginDestExpr> LastDestDecl;
  std::optional<OriginID> LastOrigin;
};

struct AssignmentQueryContext {
  const LoanPropagationAnalysis &LoanPropagation;
  const MovedLoansAnalysis &MovedLoans;
  const LiveOriginsAnalysis &LiveOrigins;
  FactManager &FactMgr;
  AnalysisDeclContext &ADC;
};

/// Retrieves a list of assignment chains for use-after-scope analysis.
///
/// To help users understand the data flow, we track where the problematic
/// address originated.
void getAliasList(const AssignmentQueryContext &Context,
                  llvm::SmallVectorImpl<AssignmentPair> &AssignmentList,
                  const Fact *CausingFact, const LoanID End,
                  const CFGBlock *StartBlock, const Expr *IssueExpr);
} // namespace clang::lifetimes::internal

#endif
