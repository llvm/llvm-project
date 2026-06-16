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
#include "llvm/ADT/ArrayRef.h"
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

  class LetterEntityBiMap {
    std::map<char, EntityId> Forward;
    std::map<EntityId, char> Reverse;

  public:
    void insert(char C, EntityId Id) {
      Forward.try_emplace(C, Id);
      Reverse[Id] = C;
    }

    EntityId operator[](char C) const { return Forward.at(C); }
    char operator[](EntityId Id) const { return Reverse.at(Id); }
    size_t size() const { return Forward.size(); }
  };

  /// Create entities for the entity domain \p EntDom in \p LU. For simplicity,
  /// entities are given by letters in \p EntDom.  Return a "bi-directional map"
  /// between letters and EntityIds.
  LetterEntityBiMap createEntities(LUSummary &LU, llvm::ArrayRef<char> EntDom) {
    LetterEntityBiMap Result;
    for (char Name : EntDom)
      Result.insert(Name, addEntity(LU, ("E" + llvm::Twine(Name)).str()));
    return Result;
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

  using Node = std::pair<char, unsigned>;
  using Edge = std::pair<Node, Node>;

  // FIXME: When we use more advanced search algorithms, it may involve
  // a divide-and-conquer approach on sub-graphs organized by contributors.
  // In that case, we may want to enumerate all possible partitions of
  // how edges are distributed among contributors. For now we use
  // `singlePartition`.

  /// Compute reachables from \p StarterLayout in the graph defined by \p
  /// EdgeLayout.  Edges and starters are all belong to one contributor.
  std::set<Node> singlePartition(llvm::ArrayRef<char> EntityDomain,
                                 llvm::ArrayRef<Edge> EdgeLayout,
                                 llvm::ArrayRef<Node> StarterLayout,
                                 unsigned Line) {
    auto LU = makeLUSummary();
    auto Entities = createEntities(*LU, EntityDomain);
    auto GetEPL = [&Entities](const Node &N) -> EntityPointerLevel {
      return buildEntityPointerLevel(Entities[N.first], N.second);
    };
    auto GetNode = [&Entities](const EntityPointerLevel &N) -> Node {
      return {Entities[N.getEntity()], N.getPointerLevel()};
    };

    std::vector<EPLEdge> Edges;
    for (const auto &[F, T] : EdgeLayout)
      Edges.push_back({GetEPL(F), GetEPL(T)});

    std::vector<EntityPointerLevel> Starters;
    for (const Node &N : StarterLayout)
      Starters.push_back(GetEPL(N));

    insertSummaries(*LU, Entities[EntityDomain[0]], Edges, Starters);
    for (size_t Idx = 1; Idx < EntityDomain.size(); ++Idx)
      insertSummaries(*LU, Entities[EntityDomain[Idx]], {}, {});

    auto Reachables = computeReachables(std::move(LU), Line);
    if (!Reachables)
      return {};

    std::set<Node> Result;
    for (auto &EPL : *Reachables)
      Result.insert(GetNode(EPL));

    return Result;
  }
};

////////////////////////////////////////////////////////////////////////////////
//  Tests below focus on pointer flow graph topology and the starter set.
//  Letters represent distinct entities; numbers represent pointer levels.
//
//  For example, `LinearChain` tests a graph forming a linear chain with 3
//  edges: (a,1) -> (b,1) -> (c,1) -> (d,1) with starter {(a,1)}.  Thus, {(a,1),
//  (b,1), (c,1), (d,1)} is the expected reachable set.
////////////////////////////////////////////////////////////////////////////////

// Linear chain: (a,1) -> (b,1) -> (c,1) -> (d,1).
// Start from {(a,1)} => {(a,1), (b,1), (c,1), (d,1)}
TEST_F(UnsafeBufferReachableAnalysisTest, LinearChain) {
  auto Reachables = singlePartition(
      /* EntityDomain */ {'a', 'b', 'c', 'd'},
      /* EdgeLayout */
      {{{'a', 1}, {'b', 1}}, {{'b', 1}, {'c', 1}}, {{'c', 1}, {'d', 1}}},
      /* StarterLayout */ {{'a', 1}}, __LINE__);
  EXPECT_EQ(Reachables.size(), 4u);
}

// Linear chain: (a,1) -> (b,2), (b,1) -> (c,2), (c,1) -> (d,2).
// Start from {(a,2)} => {(a,2), (b,3), (c,4), (d,5)}
TEST_F(UnsafeBufferReachableAnalysisTest, LinearChain2) {
  auto Reachables = singlePartition(
      /* EntityDomain */ {'a', 'b', 'c', 'd'},
      /* EdgeLayout */
      {{{'a', 1}, {'b', 2}}, {{'b', 1}, {'c', 2}}, {{'c', 1}, {'d', 2}}},
      /* StarterLayout */ {{'a', 2}}, __LINE__);
  EXPECT_EQ(Reachables.size(), 4u);
  EXPECT_EQ(Reachables,
            (std::set<Node>{{'a', 2}, {'b', 3}, {'c', 4}, {'d', 5}}));
}

// Linear chain: (a,1) -> (b,2), (b,4) -> (c,1) -> (d,1).
// Start from {(a,2)} => {(a,2), (b,3)} (halted at (b,3) — no key (b,j<=3))
TEST_F(UnsafeBufferReachableAnalysisTest, LinearChain3) {
  auto Reachables = singlePartition(
      /* EntityDomain */ {'a', 'b', 'c', 'd'},
      /* EdgeLayout */
      {{{'a', 1}, {'b', 2}}, {{'b', 4}, {'c', 1}}, {{'c', 1}, {'d', 1}}},
      /* StarterLayout */ {{'a', 2}}, __LINE__);
  EXPECT_EQ(Reachables.size(), 2u);
  EXPECT_EQ(Reachables, (std::set<Node>{{'a', 2}, {'b', 3}}));
}

// Linear chain: (a,1) -> (b,1) -> (c,1) -> (d,1).
// Start from mid-chain {(c,1)} => {(c,1), (d,1)}
TEST_F(UnsafeBufferReachableAnalysisTest, LinearChainFromMiddle) {
  auto Reachables = singlePartition(
      /* EntityDomain */ {'a', 'b', 'c', 'd'},
      /* EdgeLayout */
      {{{'a', 1}, {'b', 1}}, {{'b', 1}, {'c', 1}}, {{'c', 1}, {'d', 1}}},
      /* StarterLayout */ {{'c', 1}}, __LINE__);
  EXPECT_EQ(Reachables.size(), 2u);
  EXPECT_TRUE(Reachables.count({'c', 1}));
  EXPECT_TRUE(Reachables.count({'d', 1}));
}

// Diamond: (a,1) -> (b,1), (a,1) -> (c,1), (b,1) -> (d,1), (c,1) -> (d,1).
// Start from {(a,1)} => {(a,1), (b,1), (c,1), (d,1)}
TEST_F(UnsafeBufferReachableAnalysisTest, Diamond) {
  auto Reachables = singlePartition(
      /* EntityDomain */ {'a', 'b', 'c', 'd'},
      /* EdgeLayout */
      {{{'a', 1}, {'b', 1}},
       {{'a', 1}, {'c', 1}},
       {{'b', 1}, {'d', 1}},
       {{'c', 1}, {'d', 1}}},
      /* StarterLayout */ {{'a', 1}}, __LINE__);
  EXPECT_EQ(Reachables.size(), 4u);
}

// Diamond: (a,1) -> (b,2), (a,1) -> (c,2), (b,1) -> (d,2), (c,1) -> (d,2).
// Start from {(a,2)} => {(a,2), (b,3), (c,3), (d,4)}
TEST_F(UnsafeBufferReachableAnalysisTest, Diamond2) {
  auto Reachables = singlePartition(
      /* EntityDomain */ {'a', 'b', 'c', 'd'},
      /* EdgeLayout */
      {{{'a', 1}, {'b', 2}},
       {{'a', 1}, {'c', 2}},
       {{'b', 1}, {'d', 2}},
       {{'c', 1}, {'d', 2}}},
      /* StarterLayout */ {{'a', 2}}, __LINE__);
  EXPECT_EQ(Reachables,
            (std::set<Node>{{'a', 2}, {'b', 3}, {'c', 3}, {'d', 4}}));
}

// DisconnectedDiamond: (a,1) -> (b,2), (a,1) -> (c,2), (b,5) -> (d,1), (c,5) ->
// (d,1). Start from {(a,2)} => {(a,2), (b,3), (c,3)}
TEST_F(UnsafeBufferReachableAnalysisTest, DisconnectedDiamond) {
  auto Reachables = singlePartition(
      /* EntityDomain */ {'a', 'b', 'c', 'd'},
      /* EdgeLayout */
      {{{'a', 1}, {'b', 2}},
       {{'a', 1}, {'c', 2}},
       {{'b', 5}, {'d', 1}},
       {{'c', 5}, {'d', 1}}},
      /* StarterLayout */ {{'a', 2}}, __LINE__);
  EXPECT_EQ(Reachables, (std::set<Node>{{'a', 2}, {'b', 3}, {'c', 3}}));
}

// Diamond: (a,1) -> (b,1), (a,1) -> (c,1), (b,1) -> (d,1), (c,1) -> (d,1).
// Start from one branch {(b,1)} => {(b,1), (d,1)}
TEST_F(UnsafeBufferReachableAnalysisTest, DiamondFromBranch) {
  auto Reachables = singlePartition(
      /* EntityDomain */ {'a', 'b', 'c', 'd'},
      /* EdgeLayout */
      {{{'a', 1}, {'b', 1}},
       {{'a', 1}, {'c', 1}},
       {{'b', 1}, {'d', 1}},
       {{'c', 1}, {'d', 1}}},
      /* StarterLayout */ {{'b', 1}}, __LINE__);
  EXPECT_EQ(Reachables.size(), 2u);
  EXPECT_TRUE(Reachables.count({'b', 1}));
  EXPECT_TRUE(Reachables.count({'d', 1}));
}

// Disconnected subgraphs: (a,1) -> (b,1), (c,1) -> (d,1).
// Start from {(a,1)} => {(a,1), (b,1)}
TEST_F(UnsafeBufferReachableAnalysisTest, DisconnectedSubgraphs) {
  auto Reachables = singlePartition(
      /* EntityDomain */ {'a', 'b', 'c', 'd'},
      /* EdgeLayout */ {{{'a', 1}, {'b', 1}}, {{'c', 1}, {'d', 1}}},
      /* StarterLayout */ {{'a', 1}}, __LINE__);
  EXPECT_EQ(Reachables.size(), 2u);
  EXPECT_TRUE(Reachables.count({'a', 1}));
  EXPECT_TRUE(Reachables.count({'b', 1}));
}

// Disconnected subgraphs: (a,1) -> (b,1), (c,1) -> (d,1).
// Start from tail {(b,1)} => {(b,1)}
TEST_F(UnsafeBufferReachableAnalysisTest, DisconnectedSubgraphs2) {
  auto Reachables = singlePartition(
      /* EntityDomain */ {'a', 'b', 'c', 'd'},
      /* EdgeLayout */ {{{'a', 1}, {'b', 1}}, {{'c', 1}, {'d', 1}}},
      /* StarterLayout */ {{'b', 1}}, __LINE__);
  EXPECT_EQ(Reachables.size(), 1u);
  EXPECT_TRUE(Reachables.count({'b', 1}));
}

// Cycle: (a,1) -> (b,1) -> (c,1) -> (d,1) -> (a,1).
// Start from {(c,1)} => {(a,1), (b,1), (c,1), (d,1)}
TEST_F(UnsafeBufferReachableAnalysisTest, Cycle) {
  auto Reachables = singlePartition(
      /* EntityDomain */ {'a', 'b', 'c', 'd'},
      /* EdgeLayout */
      {{{'a', 1}, {'b', 1}},
       {{'b', 1}, {'c', 1}},
       {{'c', 1}, {'d', 1}},
       {{'d', 1}, {'a', 1}}},
      /* StarterLayout */ {{'c', 1}}, __LINE__);
  EXPECT_EQ(Reachables.size(), 4u);
  EXPECT_TRUE(Reachables.count({'a', 1}));
  EXPECT_TRUE(Reachables.count({'b', 1}));
  EXPECT_TRUE(Reachables.count({'c', 1}));
  EXPECT_TRUE(Reachables.count({'d', 1}));
}

// Cycle: (a,1) -> (b,1) -> (c,1) -> (d,1) -> (a,1).
// Start from {(c,2)} => {(a,2), (b,2), (c,2), (d,2)}
TEST_F(UnsafeBufferReachableAnalysisTest, Cycle2) {
  auto Reachables = singlePartition(
      /* EntityDomain */ {'a', 'b', 'c', 'd'},
      /* EdgeLayout */
      {{{'a', 1}, {'b', 1}},
       {{'b', 1}, {'c', 1}},
       {{'c', 1}, {'d', 1}},
       {{'d', 1}, {'a', 1}}},
      /* StarterLayout */ {{'c', 2}}, __LINE__);
  EXPECT_EQ(Reachables,
            (std::set<Node>{{'a', 2}, {'b', 2}, {'c', 2}, {'d', 2}}));
}

// Cycle: (a,1) -> (b,2) -> (c,3) -> (d,4) -> (a,1).
// Start from {(a,2)} => {(a,2), (b,3), (c,4), (d,5)}
TEST_F(UnsafeBufferReachableAnalysisTest, Cycle3) {
  auto Reachables = singlePartition(
      /* EntityDomain */ {'a', 'b', 'c', 'd'},
      /* EdgeLayout */
      {{{'a', 1}, {'b', 2}},
       {{'b', 2}, {'c', 3}},
       {{'c', 3}, {'d', 4}},
       {{'d', 4}, {'a', 1}}},
      /* StarterLayout */ {{'a', 2}}, __LINE__);
  EXPECT_EQ(Reachables,
            (std::set<Node>{{'a', 2}, {'b', 3}, {'c', 4}, {'d', 5}}));
}

// Empty graph: no edges, start from {(a,1)} => {(a,1)}
TEST_F(UnsafeBufferReachableAnalysisTest, EmptyGraph) {
  auto Reachables = singlePartition(
      /* EntityDomain */ {'a'},
      /* EdgeLayout */ {},
      /* StarterLayout */ {{'a', 1}}, __LINE__);
  EXPECT_EQ(Reachables.size(), 1u);
  EXPECT_TRUE(Reachables.count({'a', 1}));
}

// Star: (a,1) -> (b,1), (a,1) -> (c,1), (a,1) -> (d,1).
// Start from {(a,1)} => {(a,1), (b,1), (c,1), (d,1)}
TEST_F(UnsafeBufferReachableAnalysisTest, StarFromHub) {
  auto Reachables = singlePartition(
      /* EntityDomain */ {'a', 'b', 'c', 'd'},
      /* EdgeLayout */
      {{{'a', 1}, {'b', 1}}, {{'a', 1}, {'c', 1}}, {{'a', 1}, {'d', 1}}},
      /* StarterLayout */ {{'a', 1}}, __LINE__);
  EXPECT_EQ(Reachables.size(), 4u);
}

// Star: (a,1) -> (b,2), (a,1) -> (c,2), (a,1) -> (d,2).
// Start from {(a,2)} => {(a,2), (b,3), (c,3), (d,3)}
TEST_F(UnsafeBufferReachableAnalysisTest, StarFromHub2) {
  auto Reachables = singlePartition(
      /* EntityDomain */ {'a', 'b', 'c', 'd'},
      /* EdgeLayout */
      {{{'a', 1}, {'b', 2}}, {{'a', 1}, {'c', 2}}, {{'a', 1}, {'d', 2}}},
      /* StarterLayout */ {{'a', 2}}, __LINE__);
  EXPECT_EQ(Reachables,
            (std::set<Node>{{'a', 2}, {'b', 3}, {'c', 3}, {'d', 3}}));
}

// Star: (a,2) -> (b,1), (a,2) -> (c,1), (a,2) -> (d,1).
// Start from {(a,1)} => {(a,1)}
TEST_F(UnsafeBufferReachableAnalysisTest, StarFromHub3) {
  auto Reachables = singlePartition(
      /* EntityDomain */ {'a', 'b', 'c', 'd'},
      /* EdgeLayout */
      {{{'a', 2}, {'b', 1}}, {{'a', 2}, {'c', 1}}, {{'a', 2}, {'d', 1}}},
      /* StarterLayout */ {{'a', 1}}, __LINE__);
  EXPECT_EQ(Reachables, (std::set<Node>{{'a', 1}}));
}

// Star: (a,1) -> (b,1), (a,1) -> (c,1), (a,1) -> (d,1).
// Start from leaf {(c,1)} => {(c,1)}
TEST_F(UnsafeBufferReachableAnalysisTest, StarFromLeaf) {
  auto Reachables = singlePartition(
      /* EntityDomain */ {'a', 'b', 'c', 'd'},
      /* EdgeLayout */
      {{{'a', 1}, {'b', 1}}, {{'a', 1}, {'c', 1}}, {{'a', 1}, {'d', 1}}},
      /* StarterLayout */ {{'c', 1}}, __LINE__);
  EXPECT_EQ(Reachables.size(), 1u);
  EXPECT_TRUE(Reachables.count({'c', 1}));
}

// Reverse star: (a,1) -> (d,1), (b,1) -> (d,1), (c,1) -> (d,1).
// Start from {(a,1)} => {(a,1), (d,1)}
TEST_F(UnsafeBufferReachableAnalysisTest, ReverseStarFromSource) {
  auto Reachables = singlePartition(
      /* EntityDomain */ {'a', 'b', 'c', 'd'},
      /* EdgeLayout */
      {{{'a', 1}, {'d', 1}}, {{'b', 1}, {'d', 1}}, {{'c', 1}, {'d', 1}}},
      /* StarterLayout */ {{'a', 1}}, __LINE__);
  EXPECT_EQ(Reachables.size(), 2u);
  EXPECT_TRUE(Reachables.count({'a', 1}));
  EXPECT_TRUE(Reachables.count({'d', 1}));
}

// Reverse star: (a,1) -> (d,2), (b,1) -> (d,2), (c,1) -> (d,2).
// Start from {(a,2)} => {(a,2), (d,3)}
TEST_F(UnsafeBufferReachableAnalysisTest, ReverseStarFromSource2) {
  auto Reachables = singlePartition(
      /* EntityDomain */ {'a', 'b', 'c', 'd'},
      /* EdgeLayout */
      {{{'a', 1}, {'d', 2}}, {{'b', 1}, {'d', 2}}, {{'c', 1}, {'d', 2}}},
      /* StarterLayout */ {{'a', 2}}, __LINE__);
  EXPECT_EQ(Reachables, (std::set<Node>{{'a', 2}, {'d', 3}}));
}

// Reverse star: (a,1) -> (d,1), (b,1) -> (d,1), (c,1) -> (d,1).
// Start from sink {(d,1)} => {(d,1)}
TEST_F(UnsafeBufferReachableAnalysisTest, ReverseStarFromSink) {
  auto Reachables = singlePartition(
      /* EntityDomain */ {'a', 'b', 'c', 'd'},
      /* EdgeLayout */
      {{{'a', 1}, {'d', 1}}, {{'b', 1}, {'d', 1}}, {{'c', 1}, {'d', 1}}},
      /* StarterLayout */ {{'d', 1}}, __LINE__);
  EXPECT_EQ(Reachables.size(), 1u);
  EXPECT_TRUE(Reachables.count({'d', 1}));
}

// Reverse star: (a,1) -> (d,1), (b,1) -> (d,1), (c,1) -> (d,1).
// Start from sink {(d,2)} => {(d,2)}
TEST_F(UnsafeBufferReachableAnalysisTest, ReverseStarFromSink2) {
  auto Reachables = singlePartition(
      /* EntityDomain */ {'a', 'b', 'c', 'd'},
      /* EdgeLayout */
      {{{'a', 1}, {'d', 1}}, {{'b', 1}, {'d', 1}}, {{'c', 1}, {'d', 1}}},
      /* StarterLayout */ {{'d', 2}}, __LINE__);
  EXPECT_EQ(Reachables, (std::set<Node>{{'d', 2}}));
}

// Self-loop: (a,1) -> (b,1) -> (b,1) -> (c,1) -> (d,1).
// Start from {(a,1)} => {(a,1), (b,1), (c,1), (d,1)}
TEST_F(UnsafeBufferReachableAnalysisTest, SelfLoopFromRoot) {
  auto Reachables = singlePartition(
      /* EntityDomain */ {'a', 'b', 'c', 'd'},
      /* EdgeLayout */
      {{{'a', 1}, {'b', 1}},
       {{'b', 1}, {'b', 1}},
       {{'b', 1}, {'c', 1}},
       {{'c', 1}, {'d', 1}}},
      /* StarterLayout */ {{'a', 1}}, __LINE__);
  EXPECT_EQ(Reachables.size(), 4u);
}

// Self-loop: (a,1) -> (b,1) -> (b,1) -> (c,2) -> (d,2).
// Start from {(a,2)} => {(a,2), (b,2), (c,3), (d,4)}
TEST_F(UnsafeBufferReachableAnalysisTest, SelfLoopFromRoot2) {
  auto Reachables = singlePartition(
      /* EntityDomain */ {'a', 'b', 'c', 'd'},
      /* EdgeLayout */
      {{{'a', 1}, {'b', 1}},
       {{'b', 1}, {'b', 1}},
       {{'b', 1}, {'c', 2}},
       {{'c', 1}, {'d', 2}}},
      /* StarterLayout */ {{'a', 2}}, __LINE__);
  EXPECT_EQ(Reachables,
            (std::set<Node>{{'a', 2}, {'b', 2}, {'c', 3}, {'d', 4}}));
}

// Self-loop: (a,1) -> (b,1) -> (b,1) -> (c,1) -> (d,1).
// Start from {(b,1)} => {(b,1), (c,1), (d,1)}
TEST_F(UnsafeBufferReachableAnalysisTest, SelfLoopFromLoopNode) {
  auto Reachables = singlePartition(
      /* EntityDomain */ {'a', 'b', 'c', 'd'},
      /* EdgeLayout */
      {{{'a', 1}, {'b', 1}},
       {{'b', 1}, {'b', 1}},
       {{'b', 1}, {'c', 1}},
       {{'c', 1}, {'d', 1}}},
      /* StarterLayout */ {{'b', 1}}, __LINE__);
  EXPECT_EQ(Reachables.size(), 3u);
  EXPECT_TRUE(Reachables.count({'b', 1}));
  EXPECT_TRUE(Reachables.count({'c', 1}));
  EXPECT_TRUE(Reachables.count({'d', 1}));
}

// Self-loop: (a,1) -> (b,1) -> (b,1) -> (c,2) -> (d,2).
// Start from {(b,2)} => {(b,2), (c,3), (d,4)}
TEST_F(UnsafeBufferReachableAnalysisTest, SelfLoopFromLoopNode2) {
  auto Reachables = singlePartition(
      /* EntityDomain */ {'a', 'b', 'c', 'd'},
      /* EdgeLayout */
      {{{'a', 1}, {'b', 1}},
       {{'b', 1}, {'b', 1}},
       {{'b', 1}, {'c', 2}},
       {{'c', 1}, {'d', 2}}},
      /* StarterLayout */ {{'b', 2}}, __LINE__);
  EXPECT_EQ(Reachables, (std::set<Node>{{'b', 2}, {'c', 3}, {'d', 4}}));
}

// Multiple starters: (a,1) -> (b,1), (c,1) -> (d,1) (disconnected).
// Start from {(a,1), (c,1)} => {(a,1), (b,1), (c,1), (d,1)}
TEST_F(UnsafeBufferReachableAnalysisTest, MultipleStartersBothChains) {
  auto Reachables = singlePartition(
      /* EntityDomain */ {'a', 'b', 'c', 'd'},
      /* EdgeLayout */ {{{'a', 1}, {'b', 1}}, {{'c', 1}, {'d', 1}}},
      /* StarterLayout */ {{'a', 1}, {'c', 1}}, __LINE__);
  EXPECT_EQ(Reachables.size(), 4u);
}

// Multiple starters: (a,1) -> (b,2), (c,1) -> (d,2).
// Start from {(a,2), (c,2)} => {(a,2), (b,3), (c,2), (d,3)}
TEST_F(UnsafeBufferReachableAnalysisTest, MultipleStartersBothChains2) {
  auto Reachables = singlePartition(
      /* EntityDomain */ {'a', 'b', 'c', 'd'},
      /* EdgeLayout */ {{{'a', 1}, {'b', 2}}, {{'c', 1}, {'d', 2}}},
      /* StarterLayout */ {{'a', 2}, {'c', 2}}, __LINE__);
  EXPECT_EQ(Reachables,
            (std::set<Node>{{'a', 2}, {'b', 3}, {'c', 2}, {'d', 3}}));
}

// Multiple starters: (a,1) -> (b,1), (c,1) -> (d,1) (disconnected).
// Start from leaves {(b,1), (d,1)} => {(b,1), (d,1)}
TEST_F(UnsafeBufferReachableAnalysisTest, MultipleStartersLeaves) {
  auto Reachables = singlePartition(
      /* EntityDomain */ {'a', 'b', 'c', 'd'},
      /* EdgeLayout */ {{{'a', 1}, {'b', 1}}, {{'c', 1}, {'d', 1}}},
      /* StarterLayout */ {{'b', 1}, {'d', 1}}, __LINE__);
  EXPECT_EQ(Reachables.size(), 2u);
  EXPECT_TRUE(Reachables.count({'b', 1}));
  EXPECT_TRUE(Reachables.count({'d', 1}));
}

// Multi-key, same source entity: (a,1) -> (b,1), (a,2) -> (c,1).
// Start from {(a,3)} => {(a,3), (b,3), (c,2)}
TEST_F(UnsafeBufferReachableAnalysisTest, MultipleKeysSameEntity) {
  auto Reachables = singlePartition(
      /* EntityDomain */ {'a', 'b', 'c'},
      /* EdgeLayout */ {{{'a', 1}, {'b', 1}}, {{'a', 2}, {'c', 1}}},
      /* StarterLayout */ {{'a', 3}}, __LINE__);
  EXPECT_EQ(Reachables, (std::set<Node>{{'a', 3}, {'b', 3}, {'c', 2}}));
}

} // namespace
