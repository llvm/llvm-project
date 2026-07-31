//===- UnsafeBufferUsageAnalysis.cpp - WPA for UnsafeBufferUsage ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UnsafeBufferUsageAnalysis is a noop analysis.
//
// UnsafeBufferUsageAnalysisResult is a map from EntityIds to
// EntityPointerLevelSets
//===----------------------------------------------------------------------===//

#include "clang/ScalableStaticAnalysis/Analyses/UnsafeBufferUsage/UnsafeBufferUsageAnalysis.h"
#include "SSAFAnalysesCommon.h"
#include "clang/ScalableStaticAnalysis/Analyses/EntityPointerLevel/EntityPointerLevel.h"
#include "clang/ScalableStaticAnalysis/Analyses/EntityPointerLevel/EntityPointerLevelFormat.h"
#include "clang/ScalableStaticAnalysis/Analyses/PointerFlow/PointerFlowAnalysis.h"
#include "clang/ScalableStaticAnalysis/Analyses/UnsafeBufferUsage/UnsafeBufferUsage.h"
#include "clang/ScalableStaticAnalysis/Analyses/VirtualMethodFamily/VirtualMethodFamily.h"
#include "clang/ScalableStaticAnalysis/Core/Serialization/JSONFormat.h"
#include "clang/ScalableStaticAnalysis/Core/WholeProgramAnalysis/AnalysisRegistry.h"
#include "clang/ScalableStaticAnalysis/Core/WholeProgramAnalysis/SummaryAnalysis.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/JSON.h"
#include <memory>
#include <utility>

using namespace clang::ssaf;
using namespace llvm;

namespace {

json::Object serializeUnsafeBufferUsageAnalysisResult(
    const UnsafeBufferUsageAnalysisResult &R,
    JSONFormat::EntityIdToJSONFn IdToJSON) {
  json::Object Result;

  Result[UnsafeBufferUsageAnalysisResultName] =
      entityPointerLevelMapToJSON(R.UnsafeBuffers, IdToJSON);
  return Result;
}

Expected<std::unique_ptr<AnalysisResult>>
deserializeUnsafeBufferUsageAnalysisResult(
    const json::Object &Obj, JSONFormat::EntityIdFromJSONFn IdFromJSON) {
  const json::Array *Content =
      Obj.getArray(UnsafeBufferUsageAnalysisResultName);

  if (!Content)
    return makeSawButExpectedError(Obj, "an object with a key %s",
                                   UnsafeBufferUsageAnalysisResultName.data());

  auto UnsafeBuffers = entityPointerLevelMapFromJSON(*Content, IdFromJSON);

  if (!UnsafeBuffers)
    return UnsafeBuffers.takeError();

  auto Ret = std::make_unique<UnsafeBufferUsageAnalysisResult>();

  Ret->UnsafeBuffers = std::move(*UnsafeBuffers);
  return std::move(Ret);
}

JSONFormat::AnalysisResultRegistry::Add<UnsafeBufferUsageAnalysisResult>
    RegisterUnsafeBufferUsageResultForJSON(
        serializeUnsafeBufferUsageAnalysisResult,
        deserializeUnsafeBufferUsageAnalysisResult);

class UnsafeBufferUsageAnalysis final
    : public SummaryAnalysis<UnsafeBufferUsageAnalysisResult,
                             UnsafeBufferUsageEntitySummary> {
public:
  llvm::Error add(EntityId Id,
                  const UnsafeBufferUsageEntitySummary &Summary) override {
    auto UnsafeBuffersOfEntity = getUnsafeBuffers(Summary);

    getResult().UnsafeBuffers[Id] = EntityPointerLevelSet(
        UnsafeBuffersOfEntity.begin(), UnsafeBuffersOfEntity.end());
    return llvm::Error::success();
  }
};

AnalysisRegistry::Add<UnsafeBufferUsageAnalysis>
    RegisterUnsafeBufferUsageAnalysis(
        "Whole-program unsafe buffer usage analysis");

//===----------------------------------------------------------------------===//
// UnsafeBufferReachableAnalysis---computes reachable unsafe buffer nodes
//===----------------------------------------------------------------------===//

json::Object serializeUnsafeBufferReachableAnalysisResult(
    const UnsafeBufferReachableAnalysisResult &R,
    JSONFormat::EntityIdToJSONFn IdToJSON) {
  json::Object Result;

  Result[UnsafeBufferReachableAnalysisResultName] =
      entityPointerLevelMapToJSON(R.Reachables, IdToJSON);
  return Result;
}

Expected<std::unique_ptr<AnalysisResult>>
deserializeUnsafeBufferReachableAnalysisResult(
    const json::Object &Obj, JSONFormat::EntityIdFromJSONFn IdFromJSON) {
  const json::Array *Content =
      Obj.getArray(UnsafeBufferReachableAnalysisResultName);

  if (!Content)
    return makeSawButExpectedError(
        Obj, "an object with a key %s",
        UnsafeBufferReachableAnalysisResultName.data());

  auto Reachables = entityPointerLevelMapFromJSON(*Content, IdFromJSON);

  if (!Reachables)
    return Reachables.takeError();

  auto Ret = std::make_unique<UnsafeBufferReachableAnalysisResult>();

  Ret->Reachables = std::move(*Reachables);
  return std::move(Ret);
}

JSONFormat::AnalysisResultRegistry::Add<UnsafeBufferReachableAnalysisResult>
    RegisterUnsafeBufferReachableResultForJSON(
        serializeUnsafeBufferReachableAnalysisResult,
        deserializeUnsafeBufferReachableAnalysisResult);

/// Computes all the reachable "nodes" (pointers) in a pointer flow graph from a
/// provided starter node set.  Specifically, the starter set is the unsafe
/// pointers found by `UnsafeBufferUsageAnalysis`.
///
/// After the forward DFS converges, the analysis runs a family-closure pass
/// using the `VirtualMethodFamilyAnalysisResult` to propagate reachability
/// across virtual-method override equivalence classes.
class UnsafeBufferReachableAnalysis
    : public DerivedAnalysis<
          UnsafeBufferReachableAnalysisResult, PointerFlowAnalysisResult,
          UnsafeBufferUsageAnalysisResult, VirtualMethodFamilyAnalysisResult> {

  /// BoundsPropagationGraph adds bounds propagation semantics to the
  /// pointer-flow graph, which represents the set of static pointer assignment
  /// sites collected from the source code. Consider the following example:
  ///
  /// void f(int ***p, int **q) {
  ///   *p = q;
  ///   (**p)[5] = 0;
  /// }
  ///
  /// There is one static pointer assignment thus one pointer-flow edge: (p, 2)
  /// -> (q, 1). In terms of bounds propagation, this assignment implies that if
  /// 'p' at pointer level 2 requires bounds, 'q' at pointer level 1 must also
  /// have them. Furthermore, this relationship propagates to deeper indirection
  /// levels: if 'p' at level 3 requires bounds, so does 'q' at level 2.
  ///
  /// In the example above, `(**p)` requires bounds (due to the array index),
  /// and therefore `*q` must require bounds as well.
  ///
  /// To generalize the idea, the BoundsPropagationGraph is defined as a super
  /// graph of the input pointer-flow graph by:
  ///
  ///   For each edge (src, i) -> (dest, j) in the pointer-flow graph, the
  ///   BoundsPropagationGraph has a finite set of edges
  ///   {(src, i + d) -> (dest, j + d) | 0 <= d < UB}, where UB is an upper
  ///   bound based on the maximum pointer level the pointer type can have.
  struct BoundsPropagationGraph {
  private:
    const std::map<EntityPointerLevel, EntityPointerLevelSet> &PointerFlows;

  public:
    BoundsPropagationGraph(const EdgeSet &PointerFlows)
        : PointerFlows(PointerFlows) {}

    /// Returns the EntityPointerLevelSet that are reachable from \p Src by
    /// one edge in the BoundsPropagationGraph.
    EntityPointerLevelSet getDestNodes(const EntityPointerLevel &Src) const {
      unsigned SrcPtrLv = Src.getPointerLevel();
      EntityPointerLevelSet Result;

      for (unsigned P = 1; P <= SrcPtrLv; ++P) {
        auto I = PointerFlows.find(buildEntityPointerLevel(Src.getEntity(), P));

        if (I != PointerFlows.end()) {
          unsigned Delta = SrcPtrLv - P;
          for (const auto &EPL : I->second)
            Result.insert(buildEntityPointerLevel(
                EPL.getEntity(), EPL.getPointerLevel() + Delta));
        }
      }
      return Result;
    }
  };

  std::map<EntityId, BoundsPropagationGraph> BPG;
  const VirtualMethodFamilyAnalysisResult *Family = nullptr;

  // Use pointers for efficiency. EPLs are in tree-based containers that only
  // grow. So pointers to them are stable.
  using EPLPtr = const EntityPointerLevel *;

  // Find all outgoing edges from `EPL` in the `Graph`, insert their
  // destination nodes into `Reachables`, and add newly discovered nodes to
  // `Worklist`:
  void updateReachablesWithOutgoings(EPLPtr EPL,
                                     std::vector<EPLPtr> &WorkList) {
    for (auto &[Id, SubGraph] : BPG) {
      auto R = SubGraph.getDestNodes(*EPL);

      for (const auto &Dst : R) {
        auto [It, Inserted] = getResult().Reachables[Id].insert(Dst);
        if (Inserted)
          WorkList.push_back(&*It);
      }
    }
  }

  // Run the family-closure pass: propagate reachability across virtual-method
  // override equivalence classes recorded in `Family`.
  //
  // `Family` records equivalence at *slot-entity* granularity. So if slot
  // entity `e` is equivalent to slot entity `e'`, then EPL(e, L) is equivalent
  // to EPL(e', L) at *every* level L. This pass is therefore level-preserving:
  // it reconstructs the per-level relationship by keeping the level L of the
  // reachable EPL it already holds.
  //
  // A single pass suffices: mirroring EPL(e, L) onto every member of e's
  // family already covers every (family, level) pair the input contains, and
  // the EPLs it inserts map back to the same family at the same level.
  //
  // FIXME: This pass runs after the pointer-flow DFS has converged and
  // `step()` returns false, so the EPLs discovered here are never fed back
  // through the pointer-flow graph. If closure discovers EPL(e', L) and an
  // edge EPL(e', L) -> EPL(x, L) exists, EPL(x, L) is missed. Fixing this
  // requires iterating the DFS and this pass to a fixpoint; see
  // UnsafeBufferReachableAnalysisTest.FamilyClosureDoesNotRerunDFS.
  void runFamilyClosurePass() {
    auto &Reachables = getResult().Reachables;
    const auto &RetAndParamData = Family->RetAndParamData;

    if (RetAndParamData.empty())
      return;

    struct Member {
      EntityId RetOrParamId;
      EntityId OwnerMethodId;
    };
    llvm::DenseMap<EntityId, llvm::SmallVector<Member, 2>> MembersOfFamily;
    for (const auto &[ParamId, Data] : RetAndParamData) {
      MembersOfFamily[Data.FamilyId].push_back({ParamId, Data.OwnerMethodId});
    }

    // Collect the (family, level) pairs to mirror. This has to be a snapshot
    // because the loop below inserts into `Reachables`.
    DenseSet<std::pair<EntityId, unsigned>> FamilyLevels;
    for (const EntityPointerLevelSet &EPLs : make_second_range(Reachables)) {
      for (const EntityPointerLevel &E : EPLs) {
        if (auto It = RetAndParamData.find(E.getEntity());
            It != RetAndParamData.end()) {
          FamilyLevels.insert({It->second.FamilyId, E.getPointerLevel()});
        }
      }
    }

    // A reachable EPL(e, L) makes EPL(e', L) reachable for
    // every entity e' in family(e).
    for (auto [FamilyId, Level] : FamilyLevels) {
      auto It = MembersOfFamily.find(FamilyId);
      if (It == MembersOfFamily.end())
        continue;
      for (const Member &M : It->second) {
        Reachables[M.OwnerMethodId].insert(
            buildEntityPointerLevel(M.RetOrParamId, Level));
      }
    }
  }

public:
  llvm::Error
  initialize(const PointerFlowAnalysisResult &PtrFlowGraph,
             const UnsafeBufferUsageAnalysisResult &Starter,
             const VirtualMethodFamilyAnalysisResult &Family) override {
    for (auto &[Id, SubGraph] : PtrFlowGraph.Edges)
      BPG.try_emplace(Id, BoundsPropagationGraph(SubGraph));
    this->Family = &Family;
    assert(getResult().Reachables.empty());
    getResult().Reachables.insert(Starter.begin(), Starter.end());
    return llvm::Error::success();
  }

  llvm::Expected<bool> step() override {
    auto &Reachables = getResult().Reachables;
    // Simple DFS:
    std::vector<EPLPtr> Worklist;

    for (auto &[Id, EPLs] : Reachables)
      for (auto &EPL : EPLs)
        Worklist.push_back(&EPL);

    while (!Worklist.empty()) {
      EPLPtr Node = Worklist.back();
      Worklist.pop_back();

      updateReachablesWithOutgoings(Node, Worklist);
    }

    // After the forward DFS converges, run a family-closure pass that
    // propagates reachability across virtual-method override equivalence
    // classes.
    runFamilyClosurePass();

    // The DFS is not an iterative algorithm, so stop iterating by returning
    // false.
    // FIXME: `runFamilyClosurePass()` above may have made new EPLs reachable,
    // and those are never pushed back through the pointer-flow graph. Reaching
    // a true fixpoint requires alternating the DFS and the closure pass until
    // neither makes progress; see
    // UnsafeBufferReachableAnalysisTest.FamilyClosureDoesNotRerunDFS.
    return false;
  }
};

AnalysisRegistry::Add<UnsafeBufferReachableAnalysis>
    RegisterUnsafeBufferReachableAnalysis(
        "Reachable pointers from unsafe buffer usage in pointer flow graph, "
        "family-closed across virtual method overrides");

} // namespace

namespace clang::ssaf {
// NOLINTNEXTLINE(misc-use-internal-linkage)
volatile int UnsafeBufferUsageAnalysisAnchorSource = 0;
} // namespace clang::ssaf
