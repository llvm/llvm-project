//===- MovedLoans.cpp - Moved Loans Analysis --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the MovedLoansAnalysis, a forward dataflow analysis that
// tracks which loans have been moved out of their original storage location
// at each program point.
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/Analyses/LifetimeSafety/MovedLoans.h"
#include "Dataflow.h"
#include "clang/Analysis/Analyses/LifetimeSafety/Facts.h"
#include "clang/Analysis/Analyses/LifetimeSafety/LiveOrigins.h"
#include "clang/Analysis/Analyses/LifetimeSafety/LoanPropagation.h"
#include "clang/Analysis/Analyses/LifetimeSafety/Loans.h"
#include "clang/Analysis/Analyses/LifetimeSafety/Utils.h"

namespace clang::lifetimes::internal {
namespace {
struct Lattice {
  MovedLoansMap MovedLoans = MovedLoansMap(nullptr);

  explicit Lattice(MovedLoansMap MovedLoans) : MovedLoans(std::move(MovedLoans)) {}

  Lattice() = default;

  bool operator==(const Lattice &Other) const {
    return MovedLoans.getRootWithoutRetain() ==
           Other.MovedLoans.getRootWithoutRetain();
  }
  bool operator!=(const Lattice &Other) const { return !(*this == Other); }
};

class AnalysisImpl
    : public DataflowAnalysis<AnalysisImpl, Lattice, Direction::Forward> {
public:
  AnalysisImpl(const CFG &C, AnalysisDeclContext &AC, FactManager &F,
               const LoanPropagationAnalysis &LoanPropagation,
               const LiveOriginsAnalysis &LiveOrigins,
               const LoanManager &LoanMgr,
               MovedLoansMap::Factory &MovedLoansMapFactory)
      : DataflowAnalysis(C, AC, F), LoanPropagation(LoanPropagation),
        LiveOrigins(LiveOrigins), LoanMgr(LoanMgr),
        MovedLoansMapFactory(MovedLoansMapFactory) {}

  using Base::transfer;

  StringRef getAnalysisName() const { return "MovedLoans"; }

  Lattice getInitialState() { return Lattice{}; }

  /// Merges moved loan state from different control flow paths. When a loan
  /// is moved on multiple paths, picks the lexically earliest move expression.
  Lattice join(const Lattice &A, const Lattice &B) {
    MovedLoansMap MovedLoans = utils::join(
        A.MovedLoans, B.MovedLoans, MovedLoansMapFactory,
        [](const Expr *const *MoveA, const Expr *const *MoveB) -> const Expr * {
          assert(MoveA || MoveB);
          if (!MoveA)
            return *MoveB;
          if (!MoveB)
            return *MoveA;
          if (*MoveA == *MoveB)
            return *MoveA;
          return (*MoveA)->getExprLoc() < (*MoveB)->getExprLoc() ? *MoveA
                                                                 : *MoveB;
        },
        utils::JoinKind::Asymmetric);
    return Lattice(MovedLoans);
  }

  /// Marks all live loans sharing the same access path as the moved origin as
  /// potentially moved.
  Lattice transfer(Lattice In, const MovedOriginFact &F) {
    OriginID MovedOrigin = F.getMovedOrigin();
    const LoanSet *ImmediatelyMovedLoans = LoanPropagation.getLoans(MovedOrigin, &F);
    if (!ImmediatelyMovedLoans || ImmediatelyMovedLoans->isEmpty())
      return In;

    auto IsInvalidated = [&](const AccessPath &Path) {
      for (LoanID LID : *ImmediatelyMovedLoans) {
        const Loan *MovedLoan = LoanMgr.getLoan(LID);
        if (MovedLoan->getAccessPath() == Path)
          return true;
      }
      return false;
    };
    const auto &Origins = LiveOrigins.getLiveOriginsAt(&F);
    
    auto CheckLoans = [&](OriginID O, const LoanSet &Loans) {
        if (!Origins.lookup(O)) return;
        for (LoanID LiveLoan : Loans) {
            const Loan *LiveLoanPtr = LoanMgr.getLoan(LiveLoan);
            if (IsInvalidated(LiveLoanPtr->getAccessPath())) {
                if (const Expr *const *Existing = In.MovedLoans.lookup(LiveLoan))
                    if (*Existing == F.getMoveExpr())
                        continue;
                In = Lattice(MovedLoansMapFactory.add(std::move(In.MovedLoans), LiveLoan, F.getMoveExpr()));
            }
        }
    };
    
    LoanPropagation.forEachOriginWithLoansAt(&F, CheckLoans);
    return In;
  }

  const MovedLoansMap &getMovedLoans(ProgramPoint P) { return getState(P).MovedLoans; }

private:
  const LoanPropagationAnalysis &LoanPropagation;
  const LiveOriginsAnalysis &LiveOrigins;
  const LoanManager &LoanMgr;
  MovedLoansMap::Factory &MovedLoansMapFactory;
};
} // namespace

class MovedLoansAnalysis::Impl final : public AnalysisImpl {
  using AnalysisImpl::AnalysisImpl;
};

MovedLoansAnalysis::MovedLoansAnalysis(
    const CFG &C, AnalysisDeclContext &AC, FactManager &F,
    const LoanPropagationAnalysis &LoanPropagation,
    const LiveOriginsAnalysis &LiveOrigins, const LoanManager &LoanMgr,
    MovedLoansMap::Factory &MovedLoansMapFactory)
    : PImpl(std::make_unique<Impl>(C, AC, F, LoanPropagation, LiveOrigins,
                                   LoanMgr, MovedLoansMapFactory)) {
  PImpl->run();
}

MovedLoansAnalysis::~MovedLoansAnalysis() = default;

const MovedLoansMap &MovedLoansAnalysis::getMovedLoans(ProgramPoint P) const {
  return PImpl->getMovedLoans(P);
}
} // namespace clang::lifetimes::internal
