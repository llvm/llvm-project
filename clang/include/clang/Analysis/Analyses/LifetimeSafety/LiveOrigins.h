//===- LiveOrigins.h - Live Origins Analysis -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the LiveOriginAnalysis, a backward dataflow analysis that
// determines which origins are "live" at each program point. An origin is
// "live" at a program point if there's a potential future use of the pointer it
// represents. Liveness is "generated" by a read of origin's loan set (e.g., a
// `UseFact`) and is "killed" (i.e., it stops being live) when its loan set is
// overwritten (e.g. a OriginFlow killing the destination origin).
//
// This information is used for detecting use-after-free errors, as it allows us
// to check if a live origin holds a loan to an object that has already expired.
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_CLANG_ANALYSIS_ANALYSES_LIFETIMESAFETY_LIVE_ORIGINS_H
#define LLVM_CLANG_ANALYSIS_ANALYSES_LIFETIMESAFETY_LIVE_ORIGINS_H

#include "clang/Analysis/Analyses/LifetimeSafety/Dataflow.h"
#include "clang/Analysis/Analyses/LifetimeSafety/Facts.h"
#include "clang/Analysis/Analyses/LifetimeSafety/Origins.h"
#include "clang/Analysis/AnalysisDeclContext.h"
#include "clang/Analysis/CFG.h"
#include "llvm/ADT/FoldingSet.h"
#include "llvm/ADT/ImmutableMap.h"
#include "llvm/Support/Debug.h"

namespace clang::lifetimes {
namespace internal {

using OriginSet = llvm::ImmutableSet<OriginID>;

enum class LivenessKind : uint8_t {
  Dead,  // Not alive
  Maybe, // Live on some path but not all paths (may-be-live)
  Must   // Live on all paths (must-be-live)
};

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

  void dump(llvm::raw_ostream &OS, const OriginManager &OM) const;
};

/// The analysis that tracks which origins are live, with granular information
/// about the causing use fact and confidence level. This is a backward
/// analysis.
class LiveOriginAnalysis
    : public DataflowAnalysis<LiveOriginAnalysis, LivenessLattice,
                              Direction::Backward> {

public:
  LiveOriginAnalysis(const CFG &C, AnalysisDeclContext &AC, FactManager &F,
                     LivenessMap::Factory &SF)
      : DataflowAnalysis(C, AC, F), FactMgr(F), Factory(SF) {}
  using DataflowAnalysis<LiveOriginAnalysis, Lattice,
                         Direction::Backward>::transfer;

  StringRef getAnalysisName() const { return "LiveOrigins"; }

  Lattice getInitialState() { return Lattice(Factory.getEmptyMap()); }

  Lattice join(Lattice L1, Lattice L2) const;

  Lattice transfer(Lattice In, const UseFact &UF);
  Lattice transfer(Lattice In, const IssueFact &IF);
  Lattice transfer(Lattice In, const OriginFlowFact &OF);

  LivenessMap getLiveOrigins(ProgramPoint P) const {
    return getState(P).LiveOrigins;
  }

  // Dump liveness values on all test points in the program.
  void dump(llvm::raw_ostream &OS,
            llvm::StringMap<ProgramPoint> TestPoints) const;

private:
  FactManager &FactMgr;
  LivenessMap::Factory &Factory;
};
} // namespace internal
} // namespace clang::lifetimes

#endif // LLVM_CLANG_ANALYSIS_ANALYSES_LIFETIMESAFETY_LIVE_ORIGINS_H
