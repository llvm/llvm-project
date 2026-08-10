//===- LiveOrigins.cpp - Live Origins Analysis -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/Analyses/LifetimeSafety/LiveOrigins.h"
#include "Dataflow.h"
#include "clang/Analysis/Analyses/LifetimeSafety/Facts.h"
#include "llvm/Support/ErrorHandling.h"

namespace clang::lifetimes::internal {
namespace {

/// The dataflow lattice for origin liveness analysis.
/// It tracks which origins are live, why they're live (which UseFact),
/// and the confidence level of that liveness.
struct Lattice {
  /// Origins referenced from more than one block. Participates in joins.
  LivenessMap Persistent;
  /// Origins confined to a single block. Discarded at block boundaries.
  LivenessMap BlockLocal;

  Lattice() : Persistent(nullptr), BlockLocal(nullptr) {};

  Lattice(LivenessMap Persistent, LivenessMap BlockLocal)
      : Persistent(Persistent), BlockLocal(BlockLocal) {}

  bool operator==(const Lattice &Other) const {
    return Persistent == Other.Persistent && BlockLocal == Other.BlockLocal;
  }

  bool operator!=(const Lattice &Other) const { return !(*this == Other); }

  void dump(llvm::raw_ostream &OS, const OriginManager &OM) const {
    if (Persistent.isEmpty() && BlockLocal.isEmpty())
      OS << "  <empty>\n";
    for (const LivenessMap &Live : {Persistent, BlockLocal})
      for (const auto &Entry : Live) {
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

static SourceLocation GetFactLoc(CausingFactType F) {
  if (const auto *UF = F.dyn_cast<const UseFact *>())
    return UF->getUseExpr()->getExprLoc();
  if (const auto *OEF = F.dyn_cast<const OriginEscapesFact *>()) {
    if (auto *ReturnEsc = dyn_cast<ReturnEscapeFact>(OEF))
      return ReturnEsc->getReturnExpr()->getExprLoc();
    if (auto *FieldEsc = dyn_cast<FieldEscapeFact>(OEF))
      return FieldEsc->getFieldDecl()->getLocation();
    if (auto *GlobalEsc = dyn_cast<GlobalEscapeFact>(OEF))
      return GlobalEsc->getGlobal()->getLocation();
  }
  llvm_unreachable("unhandled causing fact in PointerUnion");
}

/// The analysis that tracks which origins are live, with granular information
/// about the causing use fact and confidence level. This is a backward
/// analysis.
class AnalysisImpl
    : public DataflowAnalysis<AnalysisImpl, Lattice, Direction::Backward> {

public:
  AnalysisImpl(const CFG &C, AnalysisDeclContext &AC, FactManager &F,
               LivenessMap::Factory &SF)
      : DataflowAnalysis(C, AC, F), FactMgr(F), Factory(SF),
        PersistentOrigins(F.getPersistentOrigins()) {}
  using DataflowAnalysis<AnalysisImpl, Lattice, Direction::Backward>::transfer;

  StringRef getAnalysisName() const { return "LiveOrigins"; }

  Lattice getInitialState() {
    return Lattice(Factory.getEmptyMap(), Factory.getEmptyMap());
  }

  /// An origin referenced from a single block need not be live outside it.
  ///
  /// Loans only ever enter an origin through an `IssueFact` or an
  /// `OriginFlowFact` naming it as the destination, and both would make the
  /// origin persistent if they lived in another block. So a block-local origin
  /// holds no loans anywhere outside its own block, and every consumer of
  /// liveness intersects it with the origin's loans. Dropping these here keeps
  /// them out of the joins and out of the block-entry state comparison.
  Lattice transferAtBlockExit(Lattice L) {
    return Lattice(L.Persistent, Factory.getEmptyMap());
  }

  /// Merges two lattices by combining liveness information.
  /// When the same origin has different confidence levels, we take the lower
  /// one.
  Lattice join(Lattice L1, Lattice L2) const {
    assert(L1.BlockLocal.isEmpty() && L2.BlockLocal.isEmpty() &&
           "block-local origins must not reach a block boundary");
    // Take the earliest Fact to make the join hermetic and commutative.
    auto CombineCausingFact = [](CausingFactType A,
                                 CausingFactType B) -> CausingFactType {
      if (!A)
        return B;
      if (!B)
        return A;
      return GetFactLoc(A) < GetFactLoc(B) ? A : B;
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
        return LivenessInfo(L2->CausingFact, LivenessKind::Maybe);
      if (!L2)
        return LivenessInfo(L1->CausingFact, LivenessKind::Maybe);
      return LivenessInfo(CombineCausingFact(L1->CausingFact, L2->CausingFact),
                          CombineLivenessKind(L1->Kind, L2->Kind));
    };
    // A symmetric join is required here. If an origin is live on one branch but
    // not the other, its confidence must be demoted to `Maybe`.
    LivenessMap Joined =
        utils::join(L1.Persistent, L2.Persistent, Factory, CombineLivenessInfo,
                    utils::JoinKind::Symmetric);
    return Lattice(Joined, Factory.getEmptyMap());
  }

  /// A read operation makes the origin live with definite confidence, as it
  /// dominates this program point. A write operation kills the liveness of
  /// the origin since it overwrites the value.
  Lattice transfer(Lattice In, const UseFact &UF) {
    Lattice Out = In;
    for (const OriginList *Cur = UF.getUsedOrigins(); Cur;
         Cur = Cur->peelOuterOrigin()) {
      OriginID OID = Cur->getOuterOriginID();
      // Write kills liveness.
      if (UF.isWritten())
        Out = removeLive(Out, OID);
      else
        // Read makes origin live with definite confidence (dominates this
        // point).
        Out = addLive(Out, OID, LivenessInfo(&UF, LivenessKind::Must));
    }
    return Out;
  }

  /// An escaping origin (e.g., via return) makes the origin live with definite
  /// confidence, as it dominates this program point.
  Lattice transfer(Lattice In, const OriginEscapesFact &OEF) {
    return addLive(In, OEF.getEscapedOriginID(),
                   LivenessInfo(&OEF, LivenessKind::Must));
  }

  /// Issuing a new loan to an origin kills its liveness.
  Lattice transfer(Lattice In, const IssueFact &IF) {
    return removeLive(In, IF.getOriginID());
  }

  /// An OriginFlow kills the liveness of the destination origin if `KillDest`
  /// is true. Otherwise, it propagates liveness from destination to source.
  Lattice transfer(Lattice In, const OriginFlowFact &OF) {
    Lattice Out = In;
    OriginID Dest = OF.getDestOriginID();
    // If the destination of the flow is live, the source of the flow must also
    // be marked live before this point as its value will flow into the
    // destination.
    if (const LivenessInfo *DestInfo = lookupLive(In, Dest))
      Out = addLive(Out, OF.getSrcOriginID(), *DestInfo);
    if (OF.getKillDest())
      Out = removeLive(Out, Dest);
    return Out;
  }

  Lattice transfer(Lattice In, const KillOriginFact &F) {
    return removeLive(In, F.getKilledOrigin());
  }

  Lattice transfer(Lattice In, const ExpireFact &F) {
    if (auto OID = F.getOriginID())
      return removeLive(In, *OID);
    return In;
  }

  LiveOriginSet getLiveOriginsAt(ProgramPoint P) const {
    Lattice L = getState(P);
    return LiveOriginSet{L.Persistent, L.BlockLocal};
  }

  // Dump liveness values on all test points in the program.
  void dump(llvm::raw_ostream &OS,
            const llvm::StringMap<ProgramPoint> &TestPoints) const {
    llvm::dbgs() << "==========================================\n";
    llvm::dbgs() << getAnalysisName() << " results:\n";
    llvm::dbgs() << "==========================================\n";
    for (const auto &Entry : TestPoints) {
      OS << "TestPoint: " << Entry.getKey() << "\n";
      getState(Entry.getValue()).dump(OS, FactMgr.getOriginMgr());
    }
  }

private:
  /// Routes an origin to the half of the lattice it belongs to.
  bool isPersistent(OriginID OID) const {
    return PersistentOrigins.test(OID.Value);
  }

  Lattice addLive(Lattice L, OriginID OID, LivenessInfo Info) {
    if (isPersistent(OID))
      return Lattice(Factory.add(L.Persistent, OID, Info), L.BlockLocal);
    return Lattice(L.Persistent, Factory.add(L.BlockLocal, OID, Info));
  }

  Lattice removeLive(Lattice L, OriginID OID) {
    if (isPersistent(OID))
      return Lattice(Factory.remove(L.Persistent, OID), L.BlockLocal);
    return Lattice(L.Persistent, Factory.remove(L.BlockLocal, OID));
  }

  const LivenessInfo *lookupLive(const Lattice &L, OriginID OID) const {
    return isPersistent(OID) ? L.Persistent.lookup(OID)
                             : L.BlockLocal.lookup(OID);
  }

  FactManager &FactMgr;
  LivenessMap::Factory &Factory;
  /// Origins referenced from more than one basic block; see
  /// `FactManager::getPersistentOrigins`.
  const llvm::BitVector &PersistentOrigins;
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

LiveOriginSet LiveOriginsAnalysis::getLiveOriginsAt(ProgramPoint P) const {
  return PImpl->getLiveOriginsAt(P);
}

void LiveOriginsAnalysis::dump(
    llvm::raw_ostream &OS,
    const llvm::StringMap<ProgramPoint> &TestPoints) const {
  PImpl->dump(OS, TestPoints);
}
} // namespace clang::lifetimes::internal
