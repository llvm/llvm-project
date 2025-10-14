//===- Checker.cpp - C++ Lifetime Safety Checker ----------------*- C++ -*-===//
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

#include "clang/Analysis/Analyses/LifetimeSafety/Checker.h"
#include "clang/AST/Expr.h"
#include "clang/Analysis/Analyses/LifetimeSafety/Facts.h"
#include "clang/Analysis/Analyses/LifetimeSafety/LiveOrigins.h"
#include "clang/Analysis/Analyses/LifetimeSafety/LoanPropagation.h"
#include "clang/Analysis/Analyses/LifetimeSafety/Loans.h"
#include "clang/Analysis/Analyses/PostOrderCFGView.h"
#include "clang/Analysis/AnalysisDeclContext.h"
#include "clang/Basic/SourceLocation.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/TimeProfiler.h"

namespace clang::lifetimes::internal {

static Confidence livenessKindToConfidence(LivenessKind K) {
  switch (K) {
  case LivenessKind::Must:
    return Confidence::Definite;
  case LivenessKind::Maybe:
    return Confidence::Maybe;
  case LivenessKind::Dead:
    return Confidence::None;
  }
  llvm_unreachable("unknown liveness kind");
}

namespace {

/// Struct to store the complete context for a potential lifetime violation.
struct PendingWarning {
  SourceLocation ExpiryLoc; // Where the loan expired.
  const Expr *UseExpr;      // Where the origin holding this loan was used.
  Confidence ConfidenceLevel;
};

class LifetimeChecker {
private:
  llvm::DenseMap<LoanID, PendingWarning> FinalWarningsMap;
  const LoanPropagationAnalysis &LoanPropagation;
  const LiveOriginsAnalysis &LiveOrigins;
  const FactManager &FactMgr;
  LifetimeSafetyReporter *Reporter;

public:
  LifetimeChecker(const LoanPropagationAnalysis &LoanPropagation,
                  const LiveOriginsAnalysis &LiveOrigins, const FactManager &FM,
                  AnalysisDeclContext &ADC, LifetimeSafetyReporter *Reporter)
      : LoanPropagation(LoanPropagation), LiveOrigins(LiveOrigins), FactMgr(FM),
        Reporter(Reporter) {
    for (const CFGBlock *B : *ADC.getAnalysis<PostOrderCFGView>())
      for (const Fact *F : FactMgr.getFacts(B))
        if (const auto *EF = F->getAs<ExpireFact>())
          checkExpiry(EF);
    issuePendingWarnings();
  }

  /// Checks for use-after-free errors when a loan expires.
  ///
  /// This method examines all live origins at the expiry point and determines
  /// if any of them hold the expiring loan. If so, it creates a pending
  /// warning with the appropriate confidence level based on the liveness
  /// information. The confidence reflects whether the origin is definitely
  /// or maybe live at this point.
  ///
  /// Note: This implementation considers only the confidence of origin
  /// liveness. Future enhancements could also consider the confidence of loan
  /// propagation (e.g., a loan may only be held on some execution paths).
  void checkExpiry(const ExpireFact *EF) {
    LoanID ExpiredLoan = EF->getLoanID();
    LivenessMap Origins = LiveOrigins.getLiveOriginsAt(EF);
    Confidence CurConfidence = Confidence::None;
    const UseFact *BadUse = nullptr;
    for (auto &[OID, LiveInfo] : Origins) {
      LoanSet HeldLoans = LoanPropagation.getLoans(OID, EF);
      if (!HeldLoans.contains(ExpiredLoan))
        continue;
      // Loan is defaulted.
      Confidence NewConfidence = livenessKindToConfidence(LiveInfo.Kind);
      if (CurConfidence < NewConfidence) {
        CurConfidence = NewConfidence;
        BadUse = LiveInfo.CausingUseFact;
      }
    }
    if (!BadUse)
      return;
    // We have a use-after-free.
    Confidence LastConf = FinalWarningsMap.lookup(ExpiredLoan).ConfidenceLevel;
    if (LastConf >= CurConfidence)
      return;
    FinalWarningsMap[ExpiredLoan] = {/*ExpiryLoc=*/EF->getExpiryLoc(),
                                     /*UseExpr=*/BadUse->getUseExpr(),
                                     /*ConfidenceLevel=*/CurConfidence};
  }

  void issuePendingWarnings() {
    if (!Reporter)
      return;
    for (const auto &[LID, Warning] : FinalWarningsMap) {
      const Loan &L = FactMgr.getLoanMgr().getLoan(LID);
      const Expr *IssueExpr = L.IssueExpr;
      Reporter->reportUseAfterFree(IssueExpr, Warning.UseExpr,
                                   Warning.ExpiryLoc, Warning.ConfidenceLevel);
    }
  }
};
} // namespace

void runLifetimeChecker(const LoanPropagationAnalysis &LP,
                        const LiveOriginsAnalysis &LO,
                        const FactManager &FactMgr, AnalysisDeclContext &ADC,
                        LifetimeSafetyReporter *Reporter) {
  llvm::TimeTraceScope TimeProfile("LifetimeChecker");
  LifetimeChecker Checker(LP, LO, FactMgr, ADC, Reporter);
}

} // namespace clang::lifetimes::internal
