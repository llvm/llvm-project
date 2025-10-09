// TODO: Add file header
#include "clang/Analysis/Analyses/LifetimeSafety/LiveOrigins.h"
#include "llvm/Support/ErrorHandling.h"

namespace clang::lifetimes::internal {

using Lattice = LiveOriginAnalysis::Lattice;

void LivenessLattice::dump(llvm::raw_ostream &OS,
                           const OriginManager &OM) const {
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

/// Merges two lattices by combining liveness information.
/// When the same origin has different confidence levels, we take the lower
/// one.
Lattice LiveOriginAnalysis::join(Lattice L1, Lattice L2) const {
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
Lattice LiveOriginAnalysis::transfer(Lattice In, const UseFact &UF) {
  OriginID OID = UF.getUsedOrigin(FactMgr.getOriginMgr());
  // Write kills liveness.
  if (UF.isWritten())
    return Lattice(Factory.remove(In.LiveOrigins, OID));
  // Read makes origin live with definite confidence (dominates this point).
  return Lattice(
      Factory.add(In.LiveOrigins, OID, LivenessInfo(&UF, LivenessKind::Must)));
}

/// Issuing a new loan to an origin kills its liveness.
Lattice LiveOriginAnalysis::transfer(Lattice In, const IssueFact &IF) {
  return Lattice(Factory.remove(In.LiveOrigins, IF.getOriginID()));
}

/// An OriginFlow kills the liveness of the destination origin if `KillDest`
/// is true. Otherwise, it propagates liveness from destination to source.
Lattice LiveOriginAnalysis::transfer(Lattice In, const OriginFlowFact &OF) {
  if (!OF.getKillDest())
    return In;
  return Lattice(Factory.remove(In.LiveOrigins, OF.getDestOriginID()));
}

void LiveOriginAnalysis::dump(llvm::raw_ostream &OS,
                              llvm::StringMap<ProgramPoint> TestPoints) const {
  llvm::dbgs() << "==========================================\n";
  llvm::dbgs() << getAnalysisName() << " results:\n";
  llvm::dbgs() << "==========================================\n";
  for (const auto &Entry : TestPoints) {
    OS << "TestPoint: " << Entry.getKey() << "\n";
    getState(Entry.getValue()).dump(OS, FactMgr.getOriginMgr());
  }
}

} // namespace clang::lifetimes::internal
