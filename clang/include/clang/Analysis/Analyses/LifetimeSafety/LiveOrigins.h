//===- LiveOrigins.h - Live Origins Analysis -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the LiveOriginAnalysis, a backward dataflow analysis that
// determines which origins are "live" at each program point. An origin is live
// if there's a potential future use of the pointer it represents. This
// information is used to detect use-after-free errors by checking if live
// origins hold loans to objects that have already expired.
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_CLANG_ANALYSIS_ANALYSES_LIFETIMESAFETY_LIVE_ORIGINS_H
#define LLVM_CLANG_ANALYSIS_ANALYSES_LIFETIMESAFETY_LIVE_ORIGINS_H

#include "clang/AST/Decl.h"
#include "clang/AST/Expr.h"
#include "clang/AST/Type.h"
#include "clang/Analysis/Analyses/LifetimeSafety/Dataflow.h"
#include "clang/Analysis/Analyses/LifetimeSafety/Facts.h"
#include "clang/Analysis/Analyses/LifetimeSafety/Origins.h"
#include "clang/Analysis/Analyses/LifetimeSafety/Utils.h"
#include "clang/Analysis/AnalysisDeclContext.h"
#include "clang/Analysis/CFG.h"
#include "llvm/ADT/FoldingSet.h"
#include "llvm/ADT/ImmutableMap.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"

namespace clang::lifetimes {
namespace internal {

using OriginSet = llvm::ImmutableSet<OriginID>;

enum class LivenessKind : uint8_t {
  Dead,  // Not alive
  Maybe, // Live on some path but not all paths (may-be-live)
  Must   // Live on all paths (must-be-live)
};

// ========================================================================= //
//                         Live Origins Analysis
// ========================================================================= //
//
// A backward dataflow analysis that determines which origins are "live" at each
// program point. An origin is "live" at a program point if there's a potential
// future use of the pointer it represents. Liveness is "generated" by a read of
// origin's loan set (e.g., a `UseFact`) and is "killed" (i.e., it stops being
// live) when its loan set is overwritten (e.g. a OriginFlow killing the
// destination origin).
//
// This information is used for detecting use-after-free errors, as it allows us
// to check if a live origin holds a loan to an object that has already expired.
// ========================================================================= //

/// Information about why an origin is live at a program point.
struct LivenessInfo {
  /// The use that makes the origin live. If liveness is propagated from
  /// multiple uses along different paths, this will point to the use appearing
  /// earlier in the translation unit.
  /// This is 'null' when the origin is not live.
  const UseFact *CausingUseFact;
  /// The kind of liveness of the origin.
  /// `Must`: The origin is live on all control-flow paths from the current
  /// point to the function's exit (i.e. the current point is dominated by a set
  /// of uses).
  /// `Maybe`: indicates it is live on some but not all paths.
  ///
  /// This determines the diagnostic's confidence level.
  /// `Must`-be-alive at expiration implies a definite use-after-free,
  /// while `Maybe`-be-alive suggests a potential one on some paths.
  LivenessKind Kind;

  LivenessInfo() : CausingUseFact(nullptr), Kind(LivenessKind::Dead) {}
  LivenessInfo(const UseFact *UF, LivenessKind K)
      : CausingUseFact(UF), Kind(K) {}

  bool operator==(const LivenessInfo &Other) const {
    return CausingUseFact == Other.CausingUseFact && Kind == Other.Kind;
  }
  bool operator!=(const LivenessInfo &Other) const { return !(*this == Other); }

  void Profile(llvm::FoldingSetNodeID &IDBuilder) const {
    IDBuilder.AddPointer(CausingUseFact);
    IDBuilder.Add(Kind);
  }
};

using LivenessMap = llvm::ImmutableMap<OriginID, LivenessInfo>;

/// The dataflow lattice for origin liveness analysis.
/// It tracks which origins are live, why they're live (which UseFact),
/// and the confidence level of that liveness.
struct LivenessLattice {
  LivenessMap LiveOrigins;

  LivenessLattice() : LiveOrigins(nullptr) {};

  explicit LivenessLattice(LivenessMap L) : LiveOrigins(L) {}

  bool operator==(const LivenessLattice &Other) const {
    return LiveOrigins == Other.LiveOrigins;
  }

  bool operator!=(const LivenessLattice &Other) const {
    return !(*this == Other);
  }

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
class LiveOriginAnalysis
    : public DataflowAnalysis<LiveOriginAnalysis, LivenessLattice,
                              Direction::Backward> {
  FactManager &FactMgr;
  LivenessMap::Factory &Factory;

public:
  LiveOriginAnalysis(const CFG &C, AnalysisDeclContext &AC, FactManager &F,
                     LivenessMap::Factory &SF)
      : DataflowAnalysis(C, AC, F), FactMgr(F), Factory(SF) {}
  using DataflowAnalysis<LiveOriginAnalysis, Lattice,
                         Direction::Backward>::transfer;

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

  LivenessMap getLiveOrigins(ProgramPoint P) const {
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
};
} // namespace internal
} // namespace clang::lifetimes

#endif // LLVM_CLANG_ANALYSIS_ANALYSES_LIFETIMESAFETY_LIVE_ORIGINS_H
