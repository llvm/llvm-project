//===- UnsafeBufferReachableAnalysisTest.cpp ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../TestFixture.h"
#include "clang/ScalableStaticAnalysisFramework/Analyses/EntityPointerLevel/EntityPointerLevel.h"
#include "clang/ScalableStaticAnalysisFramework/Analyses/PointerFlow/PointerFlow.h"
#include "clang/ScalableStaticAnalysisFramework/Analyses/PointerFlow/PointerFlowAnalysis.h"
#include "clang/ScalableStaticAnalysisFramework/Analyses/UnsafeBufferUsage/UnsafeBufferUsage.h"
#include "clang/ScalableStaticAnalysisFramework/Analyses/UnsafeBufferUsage/UnsafeBufferUsageAnalysis.h"
#include "clang/ScalableStaticAnalysisFramework/Core/EntityLinker/LUSummary.h"
#include "clang/ScalableStaticAnalysisFramework/Core/Model/BuildNamespace.h"
#include "clang/ScalableStaticAnalysisFramework/Core/Model/EntityId.h"
#include "clang/ScalableStaticAnalysisFramework/Core/Model/EntityLinkage.h"
#include "clang/ScalableStaticAnalysisFramework/Core/Model/EntityName.h"
#include "clang/ScalableStaticAnalysisFramework/Core/WholeProgramAnalysis/AnalysisDriver.h"
#include "clang/ScalableStaticAnalysisFramework/Core/WholeProgramAnalysis/WPASuite.h"
#include "llvm/ADT/Sequence.h"
#include "gtest/gtest.h"
#include <map>
#include <memory>
#include <optional>

using namespace clang;
using namespace ssaf;

namespace clang::ssaf {
extern PointerFlowEntitySummary buildPointerFlowEntitySummary(EdgeSet Edges);
extern UnsafeBufferUsageEntitySummary
    buildUnsafeBufferUsageEntitySummary(EntityPointerLevelSet);
} // namespace clang::ssaf

namespace {

class UnsafeBufferReachableAnalysisTest : public TestFixture {
protected:
  using EPLEdge = std::pair<EntityPointerLevel, EntityPointerLevel>;

  static constexpr EntityLinkage ExternalLinkage =
      EntityLinkage(EntityLinkageType::External);

  std::unique_ptr<LUSummary> makeLUSummary() {
    NestedBuildNamespace NS(
        {BuildNamespace(BuildNamespaceKind::LinkUnit, "TestLU")});
    return std::make_unique<LUSummary>(std::move(NS));
  }

  EntityId addEntity(LUSummary &LU, llvm::StringRef USR) {
    NestedBuildNamespace NS(
        {BuildNamespace(BuildNamespaceKind::LinkUnit, "TestLU")});
    EntityName Name(USR.str(), "", NS);
    EntityId Id = getIdTable(LU).getId(Name);
    getLinkageTable(LU).insert({Id, ExternalLinkage});
    return Id;
  }

  /// Insert a PointerFlowEntitySummary for an entity.
  void insertPointerFlowSummary(LUSummary &LU, EntityId Id, EdgeSet Edges) {
    getData(LU)[PointerFlowEntitySummary::summaryName()][Id] =
        std::make_unique<PointerFlowEntitySummary>(
            buildPointerFlowEntitySummary(std::move(Edges)));
  }

  /// Insert an UnsafeBufferUsageEntitySummary for an entity.
  void insertUnsafeBufferUsageSummary(LUSummary &LU, EntityId Id,
                                      EntityPointerLevelSet UnsafeBuffers) {
    getData(LU)[UnsafeBufferUsageEntitySummary::summaryName()][Id] =
        std::make_unique<UnsafeBufferUsageEntitySummary>(
            buildUnsafeBufferUsageEntitySummary(std::move(UnsafeBuffers)));
  }

  /// Create \p N entities in \p LU and return their EntityIds.
  std::vector<EntityId> createEntities(LUSummary &LU, unsigned N) {
    std::vector<EntityId> Ids;
    for (unsigned I = 0; I < N; ++I)
      Ids.push_back(addEntity(LU, ("E" + llvm::Twine(I)).str()));
    return Ids;
  }

  /// Create \p N EPLs, one per entity.
  std::vector<EntityPointerLevel>
  createEPLs(llvm::ArrayRef<EntityId> Entities) {
    std::vector<EntityPointerLevel> EPLs;
    for (const auto &Id : Entities)
      EPLs.push_back(buildEntityPointerLevel(Id, 1));
    return EPLs;
  }

  /// Insert both PointerFlow and UnsafeBufferUsage summaries for an entity
  /// from a list of edges and a list of starter EPLs.
  void insertSummaries(LUSummary &LU, EntityId Id,
                       llvm::ArrayRef<EPLEdge> EdgeList,
                       llvm::ArrayRef<EntityPointerLevel> StarterList) {
    EdgeSet Edges;
    for (const auto &[From, To] : EdgeList)
      Edges[From].insert(To);
    insertPointerFlowSummary(LU, Id, std::move(Edges));

    EntityPointerLevelSet Starters;
    for (const auto &EPL : StarterList)
      Starters.insert(EPL);
    insertUnsafeBufferUsageSummary(LU, Id, std::move(Starters));
  }

  /// Run the driver and return the flattened reachable EPL set.
  std::optional<EntityPointerLevelSet>
  computeReachables(std::unique_ptr<LUSummary> LU, unsigned Line) {
    AnalysisDriver Driver(std::move(LU));
    auto WPAOrErr =
        Driver.run<PointerFlowAnalysisResult, UnsafeBufferUsageAnalysisResult,
                   UnsafeBufferReachableAnalysisResult>();
    if (!WPAOrErr) {
      ADD_FAILURE_AT(__FILE__, Line) << llvm::toString(WPAOrErr.takeError());
      return std::nullopt;
    }
    auto ROrErr = WPAOrErr->get<UnsafeBufferReachableAnalysisResult>();
    if (!ROrErr) {
      ADD_FAILURE_AT(__FILE__, Line) << llvm::toString(ROrErr.takeError());
      return std::nullopt;
    }
    EntityPointerLevelSet Result;
    for (const auto &[Id, EPLs] : ROrErr->Reachables)
      Result.insert(EPLs.begin(), EPLs.end());
    return Result;
  }

  // FIXME: When we use more advanced search algorithms, it may involve
  // a divide-and-conquer approach on sub-graphs organized by contributors.
  // In that case, we may want to enumerate all possible partitions of
  // how edges are distributed among contributors. For now we use
  // `singlePartition`.

  /// Compute reachables from \p StarterLayout in the graph defined by \p
  /// EdgeLayout.  Edges and starters are all belong to Entity 0.
  std::optional<std::set<unsigned>>
  singlePartition(unsigned NumEnt,
                  llvm::ArrayRef<std::pair<unsigned, unsigned>> EdgeLayout,
                  llvm::ArrayRef<unsigned> StarterLayout, unsigned Line) {
    auto LU = makeLUSummary();
    auto Entities = createEntities(*LU, NumEnt);
    auto N = createEPLs(Entities);

    std::vector<EPLEdge> Edges;
    for (const auto &[F, T] : EdgeLayout)
      Edges.push_back({N[F], N[T]});

    std::vector<EntityPointerLevel> Starters;
    for (unsigned Idx : StarterLayout)
      Starters.push_back(N[Idx]);

    insertSummaries(*LU, Entities[0], Edges, Starters);
    for (unsigned Idx = 1; Idx < NumEnt; ++Idx)
      insertSummaries(*LU, Entities[Idx], {}, {});

    auto Reachables = computeReachables(std::move(LU), Line);
    if (!Reachables.has_value())
      return std::nullopt;

    std::set<unsigned> ReachableIndices;
    for (unsigned I : llvm::seq(0U, NumEnt))
      if (Reachables->count(N[I]))
        ReachableIndices.insert(I);

    return ReachableIndices;
  }
};

////////////////////////////////////////////////////////////////////////////////
//  Tests below are written in a manner that focuses on pointer flow graph
//  topology and the starter set, where numbers are used to represent distinct
//  nodes (pointers).
//  For example, `LinearChain` tests a graph forming a
//  linear chain with 4 distinct nodes: 0 -> 1 -> 2 -> 3 with a starter set {0},
//  where, for example, 0 -> 1 represents an edge where node 0 is the source and
//  node 1 is the destination. Thus, {0, 1, 2, 3} is the expected reachable set.
////////////////////////////////////////////////////////////////////////////////

// Linear chain: 0 -> 1 -> 2 -> 3.
// Start from {0} => {0, 1, 2, 3}
TEST_F(UnsafeBufferReachableAnalysisTest, LinearChain) {
  auto Reachables = singlePartition(
      /* NumEnt */ 4,
      /* EdgeLayout */ {{0, 1}, {1, 2}, {2, 3}},
      /* StarterLayout */ {0}, __LINE__);
  ASSERT_TRUE(Reachables.has_value());
  EXPECT_EQ(Reachables->size(), 4u);
}

// Linear chain: 0 -> 1 -> 2 -> 3.
// Start from mid-chain {2} => {2, 3}
TEST_F(UnsafeBufferReachableAnalysisTest, LinearChainFromMiddle) {
  auto Reachables = singlePartition(
      /* NumEnt */ 4,
      /* EdgeLayout */ {{0, 1}, {1, 2}, {2, 3}},
      /* StarterLayout */ {2}, __LINE__);
  ASSERT_TRUE(Reachables.has_value());
  EXPECT_EQ(Reachables->size(), 2u);
  EXPECT_TRUE(Reachables->count(2));
  EXPECT_TRUE(Reachables->count(3));
}

// Diamond: 0 -> 1, 0 -> 2, 1 -> 3, 2 -> 3.
// Start from {0} => {0, 1, 2, 3}
TEST_F(UnsafeBufferReachableAnalysisTest, Diamond) {
  auto Reachables = singlePartition(
      /* NumEnt */ 4,
      /* EdgeLayout */ {{0, 1}, {0, 2}, {1, 3}, {2, 3}},
      /* StarterLayout */ {0}, __LINE__);
  ASSERT_TRUE(Reachables.has_value());
  EXPECT_EQ(Reachables->size(), 4u);
}

// Diamond: 0 -> 1, 0 -> 2, 1 -> 3, 2 -> 3.
// Start from one branch {1} => {1, 3}
TEST_F(UnsafeBufferReachableAnalysisTest, DiamondFromBranch) {
  auto Reachables = singlePartition(
      /* NumEnt */ 4,
      /* EdgeLayout */ {{0, 1}, {0, 2}, {1, 3}, {2, 3}},
      /* StarterLayout */ {1}, __LINE__);
  ASSERT_TRUE(Reachables.has_value());
  EXPECT_EQ(Reachables->size(), 2u);
  EXPECT_TRUE(Reachables->count(1));
  EXPECT_TRUE(Reachables->count(3));
}

// Disconnected subgraphs: 0 -> 1, 2 -> 3.
// Start from {0} => {0, 1}
TEST_F(UnsafeBufferReachableAnalysisTest, DisconnectedSubgraphs) {
  auto Reachables = singlePartition(
      /* NumEnt */ 4,
      /* EdgeLayout */ {{0, 1}, {2, 3}},
      /* StarterLayout */ {0}, __LINE__);
  ASSERT_TRUE(Reachables.has_value());
  EXPECT_EQ(Reachables->size(), 2u);
  EXPECT_TRUE(Reachables->count(0));
  EXPECT_TRUE(Reachables->count(1));
}

// Disconnected subgraphs: 0 -> 1, 2 -> 3.
// Start from tail {1} => {1}
TEST_F(UnsafeBufferReachableAnalysisTest, DisconnectedSubgraphsFromTail) {
  auto Reachables = singlePartition(
      /* NumEnt */ 4,
      /* EdgeLayout */ {{0, 1}, {2, 3}},
      /* StarterLayout */ {1}, __LINE__);
  ASSERT_TRUE(Reachables.has_value());
  EXPECT_EQ(Reachables->size(), 1u);
  EXPECT_TRUE(Reachables->count(1));
}

// Cycle: 0 -> 1 -> 2 -> 3 -> 0.
// Start from {2} => {0, 1, 2, 3}
TEST_F(UnsafeBufferReachableAnalysisTest, Cycle) {
  auto Reachables = singlePartition(
      /* NumEnt */ 4,
      /* EdgeLayout */ {{0, 1}, {1, 2}, {2, 3}, {3, 0}},
      /* StarterLayout */ {2}, __LINE__);
  ASSERT_TRUE(Reachables.has_value());
  EXPECT_EQ(Reachables->size(), 4u);
  EXPECT_TRUE(Reachables->count(0));
  EXPECT_TRUE(Reachables->count(1));
  EXPECT_TRUE(Reachables->count(2));
  EXPECT_TRUE(Reachables->count(3));
}

// Empty graph: no edges, start from {0} => {0}
TEST_F(UnsafeBufferReachableAnalysisTest, EmptyGraph) {
  auto Reachables = singlePartition(
      /* NumEnt */ 4,
      /* EdgeLayout */ {},
      /* StarterLayout */ {0}, __LINE__);
  ASSERT_TRUE(Reachables.has_value());
  EXPECT_EQ(Reachables->size(), 1u);
  EXPECT_TRUE(Reachables->count(0));
}

// Star: 0 -> 1, 0 -> 2, 0 -> 3.
// Start from {0} => {0, 1, 2, 3}
TEST_F(UnsafeBufferReachableAnalysisTest, StarFromHub) {
  auto Reachables = singlePartition(
      /* NumEnt */ 4,
      /* EdgeLayout */ {{0, 1}, {0, 2}, {0, 3}},
      /* StarterLayout */ {0}, __LINE__);
  ASSERT_TRUE(Reachables.has_value());
  EXPECT_EQ(Reachables->size(), 4u);
}

// Star: 0 -> 1, 0 -> 2, 0 -> 3.
// Start from leaf {2} => {2}
TEST_F(UnsafeBufferReachableAnalysisTest, StarFromLeaf) {
  auto Reachables = singlePartition(
      /* NumEnt */ 4,
      /* EdgeLayout */ {{0, 1}, {0, 2}, {0, 3}},
      /* StarterLayout */ {2}, __LINE__);
  ASSERT_TRUE(Reachables.has_value());
  EXPECT_EQ(Reachables->size(), 1u);
  EXPECT_TRUE(Reachables->count(2));
}

// Reverse star: 0 -> 3, 1 -> 3, 2 -> 3.
// Start from {0} => {0, 3}
TEST_F(UnsafeBufferReachableAnalysisTest, ReverseStarFromSource) {
  auto Reachables = singlePartition(
      /* NumEnt */ 4,
      /* EdgeLayout */ {{0, 3}, {1, 3}, {2, 3}},
      /* StarterLayout */ {0}, __LINE__);
  ASSERT_TRUE(Reachables.has_value());
  EXPECT_EQ(Reachables->size(), 2u);
  EXPECT_TRUE(Reachables->count(0));
  EXPECT_TRUE(Reachables->count(3));
}

// Reverse star: 0 -> 3, 1 -> 3, 2 -> 3.
// Start from sink {3} => {3}
TEST_F(UnsafeBufferReachableAnalysisTest, ReverseStarFromSink) {
  auto Reachables = singlePartition(
      /* NumEnt */ 4,
      /* EdgeLayout */ {{0, 3}, {1, 3}, {2, 3}},
      /* StarterLayout */ {3}, __LINE__);
  ASSERT_TRUE(Reachables.has_value());
  EXPECT_EQ(Reachables->size(), 1u);
  EXPECT_TRUE(Reachables->count(3));
}

// Self-loop: 0 -> 1, 1 -> 1, 1 -> 2, 2 -> 3.
// Start from {0} => {0, 1, 2, 3}
TEST_F(UnsafeBufferReachableAnalysisTest, SelfLoopFromRoot) {
  auto Reachables = singlePartition(
      /* NumEnt */ 4,
      /* EdgeLayout */ {{0, 1}, {1, 1}, {1, 2}, {2, 3}},
      /* StarterLayout */ {0}, __LINE__);
  ASSERT_TRUE(Reachables.has_value());
  EXPECT_EQ(Reachables->size(), 4u);
}

// Self-loop: 0 -> 1, 1 -> 1, 1 -> 2, 2 -> 3.
// Start from {1} => {1, 2, 3}
TEST_F(UnsafeBufferReachableAnalysisTest, SelfLoopFromLoopNode) {
  auto Reachables = singlePartition(
      /* NumEnt */ 4,
      /* EdgeLayout */ {{0, 1}, {1, 1}, {1, 2}, {2, 3}},
      /* StarterLayout */ {1}, __LINE__);
  ASSERT_TRUE(Reachables.has_value());
  EXPECT_EQ(Reachables->size(), 3u);
  EXPECT_TRUE(Reachables->count(1));
  EXPECT_TRUE(Reachables->count(2));
  EXPECT_TRUE(Reachables->count(3));
}

// Multiple starters: 0 -> 1, 2 -> 3 (disconnected).
// Start from {0, 2} => {0, 1, 2, 3}
TEST_F(UnsafeBufferReachableAnalysisTest, MultipleStartersBothChains) {
  auto Reachables = singlePartition(
      /* NumEnt */ 4,
      /* EdgeLayout */ {{0, 1}, {2, 3}},
      /* StarterLayout */ {0, 2}, __LINE__);
  ASSERT_TRUE(Reachables.has_value());
  EXPECT_EQ(Reachables->size(), 4u);
}

// Multiple starters: 0 -> 1, 2 -> 3 (disconnected).
// Start from leaves {1, 3} => {1, 3}
TEST_F(UnsafeBufferReachableAnalysisTest, MultipleStartersLeaves) {
  auto Reachables = singlePartition(
      /* NumEnt */ 4,
      /* EdgeLayout */ {{0, 1}, {2, 3}},
      /* StarterLayout */ {1, 3}, __LINE__);
  ASSERT_TRUE(Reachables.has_value());
  EXPECT_EQ(Reachables->size(), 2u);
  EXPECT_TRUE(Reachables->count(1));
  EXPECT_TRUE(Reachables->count(3));
}

} // namespace
