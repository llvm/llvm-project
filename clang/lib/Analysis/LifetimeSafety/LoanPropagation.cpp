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
#include "llvm/ADT/SmallSet.h"
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
      case Fact::Kind::Use:
        for (const OriginList *Cur = F->getAs<UseFact>()->getUsedOrigins(); Cur;
             Cur = Cur->peelOuterOrigin())
          CheckOrigin(Cur->getOuterOriginID());
        break;
      case Fact::Kind::KillOrigin:
        CheckOrigin(F->getAs<KillOriginFact>()->getKilledOrigin());
        break;
      case Fact::Kind::OriginEscapes:
        // An escaping origin is read at the exit block but defined earlier, so
        // it spans blocks and must participate in joins.
        CheckOrigin(F->getAs<OriginEscapesFact>()->getEscapedOriginID());
        break;
      case Fact::Kind::MovedOrigin:
      case Fact::Kind::Expire:
      case Fact::Kind::TestPoint:
      case Fact::Kind::InvalidateOrigin:
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

  explicit Lattice(OriginLoanMap Persistent, OriginLoanMap BlockLocal)
      : PersistentOrigins(std::move(Persistent)),
        BlockLocalOrigins(std::move(BlockLocal)) {}
  Lattice() = default;

  bool operator==(const Lattice &Other) const {
    return PersistentOrigins.getRootWithoutRetain() ==
               Other.PersistentOrigins.getRootWithoutRetain() &&
           BlockLocalOrigins.getRootWithoutRetain() ==
               Other.BlockLocalOrigins.getRootWithoutRetain();
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
  Lattice join(const Lattice &A, const Lattice &B) {
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
    const LoanSet *Existing = isPersistent(OID)
                                  ? In.PersistentOrigins.lookup(OID)
                                  : In.BlockLocalOrigins.lookup(OID);
    if (Existing && Existing->isSingleton() && Existing->contains(LID))
      return In;
    LoanSet NewLoans = LoanSetFactory.add(LoanSetFactory.getEmptySet(), LID);
    return setLoans(std::move(In), OID, NewLoans, Existing);
  }

  /// A flow from source to destination. If `KillDest` is true, this replaces
  /// the destination's loans with the source's. Otherwise, the source's loans
  /// are merged into the destination's.
  Lattice transfer(Lattice In, const OriginFlowFact &F) {
    OriginID DestOID = F.getDestOriginID();
    OriginID SrcOID = F.getSrcOriginID();

    const LoanSet *Existing = isPersistent(DestOID)
                                  ? In.PersistentOrigins.lookup(DestOID)
                                  : In.BlockLocalOrigins.lookup(DestOID);

    LoanSet DestLoans =
        Existing ? (F.getKillDest() ? LoanSetFactory.getEmptySet() : *Existing)
                 : LoanSetFactory.getEmptySet();
                 
    const LoanSet *SrcLoans = getLoans(In, SrcOID);
    LoanSet MergedLoans = SrcLoans ? utils::join(DestLoans, *SrcLoans, LoanSetFactory) : DestLoans;

    return setLoans(std::move(In), DestOID, MergedLoans, Existing);
  }

  Lattice transfer(Lattice In, const KillOriginFact &F) {
    OriginID OID = F.getKilledOrigin();
    const LoanSet *Existing = isPersistent(OID)
                                  ? In.PersistentOrigins.lookup(OID)
                                  : In.BlockLocalOrigins.lookup(OID);
    return setLoans(std::move(In), OID, LoanSetFactory.getEmptySet(), Existing);
  }

  Lattice transfer(Lattice In, const ExpireFact &F) {
    if (auto OID = F.getOriginID()) {
      const LoanSet *Existing = isPersistent(*OID)
                                    ? In.PersistentOrigins.lookup(*OID)
                                    : In.BlockLocalOrigins.lookup(*OID);
      return setLoans(std::move(In), *OID, LoanSetFactory.getEmptySet(), Existing);
    }
    return In;
  }

  const LoanSet *getLoans(OriginID OID, ProgramPoint P) const {
    return getLoans(getState(P), OID);
  }

  void forEachOriginWithLoansAt(ProgramPoint P,
                                clang::lifetimes::internal::LoanPropagationAnalysis::LoanMatchCallback CB) const {
    const auto &PState = getState(P);
    for (const auto &[OID, Loans] : PState.PersistentOrigins)
      CB(OID, Loans);
    for (const auto &[OID, Loans] : PState.BlockLocalOrigins)
      CB(OID, Loans);
  }

  llvm::SmallVector<OriginID> buildOriginFlowChain(ProgramPoint StartPoint,
                                                   const OriginID StartOID,
                                                   const LoanID TargetLoan,
                                                   const CFG *Cfg) const {
    const LoanSet *StartLoans = getLoans(StartOID, StartPoint);
    assert(StartLoans && StartLoans->contains(TargetLoan) &&
           "TargetLoan must be present in the StartOID at the StartPoint");

    // Locate the CFG block containing the StartPoint
    const CFGBlock *EndBlock = nullptr;
    size_t BlockID = FactMgr.getBlockID(StartPoint);
    for (const CFGBlock *Block : *Cfg)
      if (Block->getBlockID() == BlockID) {
        EndBlock = Block;
        break;
      }

    // Set up DFS traversal state
    // SearchState tracks which block we're in and which origin we're tracing
    // Each DFSNode maintains its own OriginFlowChain.
    using SearchState = std::pair<const CFGBlock *, OriginID>;
    struct DFSNode {
      SearchState CurrState;
      llvm::SmallVector<OriginID> OriginFlowChain;
    };

    llvm::SmallVector<DFSNode> PendingStates;
    llvm::SmallSet<SearchState, 16> VistedStates;
    PendingStates.push_back({{EndBlock, StartOID}, {}});

    // DFS loop to trace loan backwards through CFG
    while (!PendingStates.empty()) {
      DFSNode CurrNode = PendingStates.pop_back_val();
      auto [CurrBlock, CurrOID] = CurrNode.CurrState;

      // Trace origins within the current block
      const auto [BuildResult, Complete] =
          buildOriginFlowChain(CurrBlock, CurrOID, TargetLoan);
      if (!BuildResult.empty()) {
        CurrNode.OriginFlowChain.append(BuildResult);
        CurrOID = BuildResult.back();
      }

      // If we found the IssueFact, we're done
      if (Complete)
        return CurrNode.OriginFlowChain;

      // Only explore predecessor blocks where the target loan is present in the
      // current origin.
      for (const CFGBlock *PredBlock : CurrBlock->preds()) {
        SearchState NextState = {PredBlock, CurrOID};
        auto Out = getOutState(PredBlock);
        if (Out) {
          if (const LoanSet *OutLoans = getLoans(*Out, CurrOID)) {
            if (OutLoans->contains(TargetLoan) &&
                VistedStates.insert(NextState).second)
              PendingStates.push_back({NextState, CurrNode.OriginFlowChain});
          }
        }
          PendingStates.push_back({NextState, CurrNode.OriginFlowChain});
      }
    }

    llvm_unreachable(
        "buildOriginFlowChain did not reach IssueFact for TargetLoan");
  }

  llvm::SmallVector<OriginID> buildOriginFlowChain(const UseFact *UF,
                                                   const LoanID TargetLoan,
                                                   const CFG *Cfg) const {
    for (const OriginList *Cur = UF->getUsedOrigins(); Cur;
         Cur = Cur->peelOuterOrigin())
      if (const LoanSet *Loans = getLoans(Cur->getOuterOriginID(), UF))
        if (Loans->contains(TargetLoan))
          return buildOriginFlowChain(UF, Cur->getOuterOriginID(), TargetLoan,
                                      Cfg);

    return {};
  }

private:
  /// Returns true if the origin is persistent (referenced in multiple blocks).
  bool isPersistent(OriginID OID) const {
    return PersistentOrigins.test(OID.Value);
  }

  Lattice setLoans(Lattice L, OriginID OID, LoanSet Loans, const LoanSet *Existing) {
    if (Existing && *Existing == Loans)
      return L;
      
    if (Loans.isEmpty()) {
      if (!Existing)
        return L;
      if (isPersistent(OID))
        return Lattice(OriginLoanMapFactory.remove(L.PersistentOrigins, OID),
                       std::move(L.BlockLocalOrigins));
      return Lattice(std::move(L.PersistentOrigins),
                     OriginLoanMapFactory.remove(L.BlockLocalOrigins, OID));
    }
    if (isPersistent(OID))
      return Lattice(OriginLoanMapFactory.add(L.PersistentOrigins, OID, Loans),
                     std::move(L.BlockLocalOrigins));
    return Lattice(std::move(L.PersistentOrigins),
                   OriginLoanMapFactory.add(L.BlockLocalOrigins, OID, Loans));
  }

  const LoanSet *getLoans(const Lattice &L, OriginID OID) const {
    const OriginLoanMap *Map =
        isPersistent(OID) ? &L.PersistentOrigins : &L.BlockLocalOrigins;
    return Map->lookup(OID);
  }

  /// Builds the chain of origins through which a loan has propagated.
  ///
  /// This procedure operates strictly within a single Block. Starting from the
  /// last fact of the Block, it traces backwards through OriginFlowFacts to
  /// identify the sequence of origins through which the loan flowed.
  ///
  /// Returns (chain, true) if the target loan origin is found during the
  /// traversal, otherwise returns (chain, false).
  std::pair<llvm::SmallVector<OriginID>, bool>
  buildOriginFlowChain(const CFGBlock *Block, const OriginID StartOID,
                       const LoanID TargetLoan) const {
    OriginID CurrOID = StartOID;
    llvm::SmallVector<OriginID> OriginFlowChain;

    for (const Fact *F : llvm::reverse(FactMgr.getFacts(Block))) {
      if (const auto *IF = F->getAs<IssueFact>())
        if (IF->getLoanID() == TargetLoan && IF->getOriginID() == CurrOID)
          return {OriginFlowChain, true};

      const auto *OFF = F->getAs<OriginFlowFact>();
      if (!OFF || OFF->getDestOriginID() != CurrOID)
        continue;

      const OriginID SrcOriginID = OFF->getSrcOriginID();
      const LoanSet *Loans = getLoans(SrcOriginID, OFF);
      if (!Loans || !Loans->contains(TargetLoan))
        continue;

      OriginFlowChain.push_back(SrcOriginID);
      CurrOID = SrcOriginID;
    }

    return {OriginFlowChain, false};
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

const LoanSet *LoanPropagationAnalysis::getLoans(OriginID OID, ProgramPoint P) const {
  return PImpl->getLoans(OID, P);
}

void LoanPropagationAnalysis::forEachOriginWithLoansAt(ProgramPoint P, LoanPropagationAnalysis::LoanMatchCallback CB) const {
    PImpl->forEachOriginWithLoansAt(P, CB);
}

llvm::SmallVector<OriginID> LoanPropagationAnalysis::buildOriginFlowChain(
    ProgramPoint StartPoint, const OriginID StartOID, const LoanID TargetLoan,
    const CFG *Cfg) const {
  return PImpl->buildOriginFlowChain(StartPoint, StartOID, TargetLoan, Cfg);
}

llvm::SmallVector<OriginID> LoanPropagationAnalysis::buildOriginFlowChain(
    const UseFact *UF, const LoanID TargetLoan, const CFG *Cfg) const {
  return PImpl->buildOriginFlowChain(UF, TargetLoan, Cfg);
}
} // namespace clang::lifetimes::internal
