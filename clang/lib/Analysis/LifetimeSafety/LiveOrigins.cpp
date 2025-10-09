//===- LiveOrigins.cpp - Live Origins Analysis -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/Analyses/LifetimeSafety/LiveOrigins.h"
#include "Dataflow.h"
#include "llvm/Support/ErrorHandling.h"

namespace clang::lifetimes::internal {
namespace {

/// The dataflow lattice for origin liveness analysis.
/// It tracks which origins are live, why they're live (which UseFact),
/// and the confidence level of that liveness.
struct Lattice {
  LivenessMap LiveOrigins;

  Lattice() : LiveOrigins(nullptr) {};

  explicit Lattice(LivenessMap L) : LiveOrigins(L) {}

  bool operator==(const Lattice &Other) const {
    return LiveOrigins == Other.LiveOrigins;
  }

  bool operator!=(const Lattice &Other) const { return !(*this == Other); }

  void dump(llvm::raw_ostream &OS, const OriginManager &OM) const {
    if (LiveOrigins.isEmpty())
      OS << "  <empty>\n";
    for (const auto &Entry : LiveOrigins) {
      OriginID OID = Entry.first;
      const LivenessInfo &Info = Entry.second;
      OS << "  ";
      OM.dump(OID, OS);
      OS << " is ";
      switch (Info.Kind) {
      case LivenessKind::Must:
        OS << "definitely";
        break;
      case LivenessKind::Maybe:
        OS << "maybe";
        break;
      case LivenessKind::Dead:
        llvm_unreachable("liveness kind of live origins should not be dead.");
      }
      OS << " live at this point\n";
    }
  }
};

/// The analysis that tracks which origins are live, with granular information
/// about the causing use fact and confidence level. This is a backward
/// analysis.
class AnalysisImpl
    : public DataflowAnalysis<AnalysisImpl, Lattice, Direction::Backward> {

public:
  AnalysisImpl(const CFG &C, AnalysisDeclContext &AC, FactManager &F,
               LivenessMap::Factory &SF)
      : DataflowAnalysis(C, AC, F), FactMgr(F), Factory(SF) {}
  using DataflowAnalysis<AnalysisImpl, Lattice, Direction::Backward>::transfer;

  StringRef getAnalysisName() const { return "LiveOrigins"; }

  Lattice getInitialState() { return Lattice(Factory.getEmptyMap()); }

  /// Merges two lattices by combining liveness information.
  /// When the same origin has different confidence levels, we take the lower
  /// one.
  Lattice join(Lattice L1, Lattice L2) const {
    LivenessMap Merged = L1.LiveOrigins;
    // Take the earliest UseFact to make the join hermetic and commutative.
    auto CombineUseFact = [](const UseFact &A,
                             const UseFact &B) -> const UseFact * {
      return A.getUseExpr()->getExprLoc() < B.getUseExpr()->getExprLoc() ? &A
                                                                         : &B;
    };
    auto CombineLivenessKind = [](LivenessKind K1,
                                  LivenessKind K2) -> LivenessKind {
      assert(K1 != LivenessKind::Dead && "LivenessKind should not be dead.");
      assert(K2 != LivenessKind::Dead && "LivenessKind should not be dead.");
      // Only return "Must" if both paths are "Must", otherwise Maybe.
      if (K1 == LivenessKind::Must && K2 == LivenessKind::Must)
        return LivenessKind::Must;
      return LivenessKind::Maybe;
    };
    auto CombineLivenessInfo = [&](const LivenessInfo *L1,
                                   const LivenessInfo *L2) -> LivenessInfo {
      assert((L1 || L2) && "unexpectedly merging 2 empty sets");
      if (!L1)
        return LivenessInfo(L2->CausingUseFact, LivenessKind::Maybe);
      if (!L2)
        return LivenessInfo(L1->CausingUseFact, LivenessKind::Maybe);
      return LivenessInfo(
          CombineUseFact(*L1->CausingUseFact, *L2->CausingUseFact),
          CombineLivenessKind(L1->Kind, L2->Kind));
    };
    return Lattice(utils::join(
        L1.LiveOrigins, L2.LiveOrigins, Factory, CombineLivenessInfo,
        // A symmetric join is required here. If an origin is live on one
        // branch but not the other, its confidence must be demoted to `Maybe`.
        utils::JoinKind::Symmetric));
  }

  /// A read operation makes the origin live with definite confidence, as it
  /// dominates this program point. A write operation kills the liveness of
  /// the origin since it overwrites the value.
  Lattice transfer(Lattice In, const UseFact &UF) {
    OriginID OID = UF.getUsedOrigin(FactMgr.getOriginMgr());
    // Write kills liveness.
    if (UF.isWritten())
      return Lattice(Factory.remove(In.LiveOrigins, OID));
    // Read makes origin live with definite confidence (dominates this point).
    return Lattice(Factory.add(In.LiveOrigins, OID,
                               LivenessInfo(&UF, LivenessKind::Must)));
  }

  /// Issuing a new loan to an origin kills its liveness.
  Lattice transfer(Lattice In, const IssueFact &IF) {
    return Lattice(Factory.remove(In.LiveOrigins, IF.getOriginID()));
  }

  /// An OriginFlow kills the liveness of the destination origin if `KillDest`
  /// is true. Otherwise, it propagates liveness from destination to source.
  Lattice transfer(Lattice In, const OriginFlowFact &OF) {
    if (!OF.getKillDest())
      return In;
    return Lattice(Factory.remove(In.LiveOrigins, OF.getDestOriginID()));
  }

  LivenessMap getLiveOriginsAt(ProgramPoint P) const {
    return getState(P).LiveOrigins;
  }

  // Dump liveness values on all test points in the program.
  void dump(llvm::raw_ostream &OS,
            llvm::StringMap<ProgramPoint> TestPoints) const {
    llvm::dbgs() << "==========================================\n";
    llvm::dbgs() << getAnalysisName() << " results:\n";
    llvm::dbgs() << "==========================================\n";
    for (const auto &Entry : TestPoints) {
      OS << "TestPoint: " << Entry.getKey() << "\n";
      getState(Entry.getValue()).dump(OS, FactMgr.getOriginMgr());
    }
  }

private:
  FactManager &FactMgr;
  LivenessMap::Factory &Factory;
};
} // namespace

// PImpl wrapper implementation
class LiveOriginsAnalysis::Impl : public AnalysisImpl {
  using AnalysisImpl::AnalysisImpl;
};

LiveOriginsAnalysis::LiveOriginsAnalysis(const CFG &C, AnalysisDeclContext &AC,
                                         FactManager &F,
                                         LivenessMap::Factory &SF)
    : PImpl(std::make_unique<Impl>(C, AC, F, SF)) {
  PImpl->run();
}

LiveOriginsAnalysis::~LiveOriginsAnalysis() = default;

LivenessMap LiveOriginsAnalysis::getLiveOriginsAt(ProgramPoint P) const {
  return PImpl->getLiveOriginsAt(P);
}

void LiveOriginsAnalysis::dump(llvm::raw_ostream &OS,
                               llvm::StringMap<ProgramPoint> TestPoints) const {
  PImpl->dump(OS, TestPoints);
}
} // namespace clang::lifetimes::internal
