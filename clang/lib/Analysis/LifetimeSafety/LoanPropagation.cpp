//===- LoanPropagation.cpp - Loan Propagation Analysis ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include <cassert>
#include <memory>

#include "Dataflow.h"
#include "clang/Analysis/Analyses/LifetimeSafety/Facts.h"
#include "clang/Analysis/Analyses/LifetimeSafety/LoanPropagation.h"
#include "clang/Analysis/Analyses/LifetimeSafety/Loans.h"
#include "clang/Analysis/Analyses/LifetimeSafety/Origins.h"
#include "clang/Analysis/Analyses/LifetimeSafety/Utils.h"
#include "clang/Analysis/AnalysisDeclContext.h"
#include "clang/Analysis/CFG.h"
#include "clang/Basic/LLVM.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/TimeProfiler.h"
#include "llvm/Support/raw_ostream.h"

namespace clang::lifetimes::internal {

// Prepass to find persistent origins. An origin is persistent if it is
// referenced in more than one basic block.
static llvm::BitVector computePersistentOrigins(const FactManager &FactMgr,
                                                const CFG &C) {
  llvm::TimeTraceScope("ComputePersistentOrigins");
  unsigned NumOrigins = FactMgr.getOriginMgr().getNumOrigins();
  llvm::BitVector PersistentOrigins(NumOrigins);

  llvm::SmallVector<const CFGBlock *> OriginToFirstSeenBlock(NumOrigins,
                                                             nullptr);
  for (const CFGBlock *B : C) {
    for (const Fact *F : FactMgr.getFacts(B)) {
      auto CheckOrigin = [&](OriginID OID) {
        if (PersistentOrigins.test(OID.Value))
          return;
        auto &FirstSeenBlock = OriginToFirstSeenBlock[OID.Value];
        if (FirstSeenBlock == nullptr)
          FirstSeenBlock = B;
        if (FirstSeenBlock != B) {
          // We saw this origin in more than one block.
          PersistentOrigins.set(OID.Value);
        }
      };

      switch (F->getKind()) {
      case Fact::Kind::Issue:
        CheckOrigin(F->getAs<IssueFact>()->getOriginID());
        break;
      case Fact::Kind::OriginFlow: {
        const auto *OF = F->getAs<OriginFlowFact>();
        CheckOrigin(OF->getDestOriginID());
        CheckOrigin(OF->getSrcOriginID());
        break;
      }
      case Fact::Kind::ReturnOfOrigin:
        CheckOrigin(F->getAs<ReturnOfOriginFact>()->getReturnedOriginID());
        break;
      case Fact::Kind::Use:
        CheckOrigin(F->getAs<UseFact>()->getUsedOrigin());
        break;
      case Fact::Kind::Expire:
      case Fact::Kind::TestPoint:
        break;
      }
    }
  }
  return PersistentOrigins;
}

namespace {

/// Represents the dataflow lattice for loan propagation.
///
/// This lattice tracks which loans each origin may hold at a given program
/// point.The lattice has a finite height: An origin's loan set is bounded by
/// the total number of loans in the function.
struct Lattice {
  /// The map from an origin to the set of loans it contains.
  /// Origins that appear in multiple blocks. Participates in join operations.
  OriginLoanMap PersistentOrigins = OriginLoanMap(nullptr);
  /// Origins confined to a single block. Discarded at block boundaries.
  OriginLoanMap BlockLocalOrigins = OriginLoanMap(nullptr);

  explicit Lattice(const OriginLoanMap &Persistent,
                   const OriginLoanMap &BlockLocal)
      : PersistentOrigins(Persistent), BlockLocalOrigins(BlockLocal) {}
  Lattice() = default;

  bool operator==(const Lattice &Other) const {
    return PersistentOrigins == Other.PersistentOrigins &&
           BlockLocalOrigins == Other.BlockLocalOrigins;
  }
  bool operator!=(const Lattice &Other) const { return !(*this == Other); }

  void dump(llvm::raw_ostream &OS) const {
    OS << "LoanPropagationLattice State:\n";
    OS << " Persistent Origins:\n";
    if (PersistentOrigins.isEmpty())
      OS << "  <empty>\n";
    for (const auto &Entry : PersistentOrigins) {
      if (Entry.second.isEmpty())
        OS << "  Origin " << Entry.first << " contains no loans\n";
      for (const LoanID &LID : Entry.second)
        OS << "  Origin " << Entry.first << " contains Loan " << LID << "\n";
    }
    OS << " Block-Local Origins:\n";
    if (BlockLocalOrigins.isEmpty())
      OS << "  <empty>\n";
    for (const auto &Entry : BlockLocalOrigins) {
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
        LoanSetFactory(LoanSetFactory),
        PersistentOrigins(computePersistentOrigins(F, C)) {}

  using Base::transfer;

  StringRef getAnalysisName() const { return "LoanPropagation"; }

  Lattice getInitialState() { return Lattice{}; }

  /// Merges two lattices by taking the union of loans for each origin.
  /// Only persistent origins are joined; block-local origins are discarded.
  Lattice join(Lattice A, Lattice B) {
    OriginLoanMap JoinedOrigins = utils::join(
        A.PersistentOrigins, B.PersistentOrigins, OriginLoanMapFactory,
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
    return Lattice(JoinedOrigins, OriginLoanMapFactory.getEmptyMap());
  }

  /// A new loan is issued to the origin. Old loans are erased.
  Lattice transfer(Lattice In, const IssueFact &F) {
    OriginID OID = F.getOriginID();
    LoanID LID = F.getLoanID();
    LoanSet NewLoans = LoanSetFactory.add(LoanSetFactory.getEmptySet(), LID);
    return setLoans(In, OID, NewLoans);
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

    return setLoans(In, DestOID, MergedLoans);
  }

  LoanSet getLoans(OriginID OID, ProgramPoint P) const {
    return getLoans(getState(P), OID);
  }

private:
  /// Returns true if the origin is persistent (referenced in multiple blocks).
  bool isPersistent(OriginID OID) const {
    return PersistentOrigins.test(OID.Value);
  }

  Lattice setLoans(Lattice L, OriginID OID, LoanSet Loans) {
    if (isPersistent(OID))
      return Lattice(OriginLoanMapFactory.add(L.PersistentOrigins, OID, Loans),
                     L.BlockLocalOrigins);
    return Lattice(L.PersistentOrigins,
                   OriginLoanMapFactory.add(L.BlockLocalOrigins, OID, Loans));
  }

  LoanSet getLoans(Lattice L, OriginID OID) const {
    const OriginLoanMap *Map =
        isPersistent(OID) ? &L.PersistentOrigins : &L.BlockLocalOrigins;
    if (auto *Loans = Map->lookup(OID))
      return *Loans;
    return LoanSetFactory.getEmptySet();
  }

  OriginLoanMap::Factory &OriginLoanMapFactory;
  LoanSet::Factory &LoanSetFactory;
  /// Boolean vector indexed by origin ID. If true, the origin appears in
  /// multiple basic blocks and must participate in join operations. If false,
  /// the origin is block-local and can be discarded at block boundaries.
  llvm::BitVector PersistentOrigins;
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
