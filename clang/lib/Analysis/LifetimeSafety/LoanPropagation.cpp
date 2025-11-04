//===- LoanPropagation.cpp - Loan Propagation Analysis ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "clang/Analysis/Analyses/LifetimeSafety/LoanPropagation.h"
#include "Dataflow.h"
#include <memory>

namespace clang::lifetimes::internal {
namespace {
/// Represents the dataflow lattice for loan propagation.
///
/// This lattice tracks which loans each origin may hold at a given program
/// point.The lattice has a finite height: An origin's loan set is bounded by
/// the total number of loans in the function.
/// TODO(opt): To reduce the lattice size, propagate origins of declarations,
/// not expressions, because expressions are not visible across blocks.
struct Lattice {
  /// The map from an origin to the set of loans it contains.
  OriginLoanMap Origins = OriginLoanMap(nullptr);

  explicit Lattice(const OriginLoanMap &S) : Origins(S) {}
  Lattice() = default;

  bool operator==(const Lattice &Other) const {
    return Origins == Other.Origins;
  }
  bool operator!=(const Lattice &Other) const { return !(*this == Other); }

  void dump(llvm::raw_ostream &OS) const {
    OS << "LoanPropagationLattice State:\n";
    if (Origins.isEmpty())
      OS << "  <empty>\n";
    for (const auto &Entry : Origins) {
      if (Entry.second.isEmpty())
        OS << "  Origin " << Entry.first << " contains no loans\n";
      for (const LoanID &LID : Entry.second)
        OS << "  Origin " << Entry.first << " contains Loan " << LID << "\n";
    }
  }
};

class AnalysisImpl
    : public DataflowAnalysis<AnalysisImpl, Lattice, Direction::Forward> {
public:
  AnalysisImpl(const CFG &C, AnalysisDeclContext &AC, FactManager &F,
               OriginLoanMap::Factory &OriginLoanMapFactory,
               LoanSet::Factory &LoanSetFactory)
      : DataflowAnalysis(C, AC, F), OriginLoanMapFactory(OriginLoanMapFactory),
        LoanSetFactory(LoanSetFactory) {}

  using Base::transfer;

  StringRef getAnalysisName() const { return "LoanPropagation"; }

  Lattice getInitialState() { return Lattice{}; }

  /// Merges two lattices by taking the union of loans for each origin.
  // TODO(opt): Keep the state small by removing origins which become dead.
  Lattice join(Lattice A, Lattice B) {
    OriginLoanMap JoinedOrigins = utils::join(
        A.Origins, B.Origins, OriginLoanMapFactory,
        [&](const LoanSet *S1, const LoanSet *S2) {
          assert((S1 || S2) && "unexpectedly merging 2 empty sets");
          if (!S1)
            return *S2;
          if (!S2)
            return *S1;
          return utils::join(*S1, *S2, LoanSetFactory);
        },
        // Asymmetric join is a performance win. For origins present only on one
        // branch, the loan set can be carried over as-is.
        utils::JoinKind::Asymmetric);
    return Lattice(JoinedOrigins);
  }

  /// A new loan is issued to the origin. Old loans are erased.
  Lattice transfer(Lattice In, const IssueFact &F) {
    OriginID OID = F.getOriginID();
    LoanID LID = F.getLoanID();
    return Lattice(OriginLoanMapFactory.add(
        In.Origins, OID,
        LoanSetFactory.add(LoanSetFactory.getEmptySet(), LID)));
  }

  /// A flow from source to destination. If `KillDest` is true, this replaces
  /// the destination's loans with the source's. Otherwise, the source's loans
  /// are merged into the destination's.
  Lattice transfer(Lattice In, const OriginFlowFact &F) {
    OriginID DestOID = F.getDestOriginID();
    OriginID SrcOID = F.getSrcOriginID();

    LoanSet DestLoans =
        F.getKillDest() ? LoanSetFactory.getEmptySet() : getLoans(In, DestOID);
    LoanSet SrcLoans = getLoans(In, SrcOID);
    LoanSet MergedLoans = utils::join(DestLoans, SrcLoans, LoanSetFactory);

    return Lattice(OriginLoanMapFactory.add(In.Origins, DestOID, MergedLoans));
  }

  LoanSet getLoans(OriginID OID, ProgramPoint P) const {
    return getLoans(getState(P), OID);
  }

private:
  LoanSet getLoans(Lattice L, OriginID OID) const {
    if (auto *Loans = L.Origins.lookup(OID))
      return *Loans;
    return LoanSetFactory.getEmptySet();
  }

  OriginLoanMap::Factory &OriginLoanMapFactory;
  LoanSet::Factory &LoanSetFactory;
};
} // namespace

class LoanPropagationAnalysis::Impl final : public AnalysisImpl {
  using AnalysisImpl::AnalysisImpl;
};

LoanPropagationAnalysis::LoanPropagationAnalysis(
    const CFG &C, AnalysisDeclContext &AC, FactManager &F,
    OriginLoanMap::Factory &OriginLoanMapFactory,
    LoanSet::Factory &LoanSetFactory)
    : PImpl(std::make_unique<Impl>(C, AC, F, OriginLoanMapFactory,
                                   LoanSetFactory)) {
  PImpl->run();
}

LoanPropagationAnalysis::~LoanPropagationAnalysis() = default;

LoanSet LoanPropagationAnalysis::getLoans(OriginID OID, ProgramPoint P) const {
  return PImpl->getLoans(OID, P);
}
} // namespace clang::lifetimes::internal
