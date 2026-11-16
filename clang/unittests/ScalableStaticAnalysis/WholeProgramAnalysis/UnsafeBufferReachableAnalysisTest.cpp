//===- UnsafeBufferReachableAnalysisTest.cpp ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../FindDecl.h"
#include "../TestFixture.h"
#include "clang/Frontend/ASTUnit.h"
#include "clang/Frontend/SSAFOptions.h"
#include "clang/ScalableStaticAnalysis/Analyses/EntityPointerLevel/EntityPointerLevel.h"
#include "clang/ScalableStaticAnalysis/Analyses/PointerFlow/PointerFlow.h"
#include "clang/ScalableStaticAnalysis/Analyses/PointerFlow/PointerFlowAnalysis.h"
#include "clang/ScalableStaticAnalysis/Analyses/TypeConstrainedPointers/TypeConstrainedPointers.h"
#include "clang/ScalableStaticAnalysis/Analyses/UnsafeBufferUsage/UnsafeBufferUsage.h"
#include "clang/ScalableStaticAnalysis/Analyses/UnsafeBufferUsage/UnsafeBufferUsageAnalysis.h"
#include "clang/ScalableStaticAnalysis/Analyses/VirtualMethodFamily/VirtualMethodFamily.h"
#include "clang/ScalableStaticAnalysis/Core/ASTEntityMapping.h"
#include "clang/ScalableStaticAnalysis/Core/EntityLinker/EntityLinker.h"
#include "clang/ScalableStaticAnalysis/Core/EntityLinker/LUSummary.h"
#include "clang/ScalableStaticAnalysis/Core/EntityLinker/LUSummaryEncoding.h"
#include "clang/ScalableStaticAnalysis/Core/EntityLinker/TUSummaryEncoding.h"
#include "clang/ScalableStaticAnalysis/Core/Model/BuildNamespace.h"
#include "clang/ScalableStaticAnalysis/Core/Model/EntityId.h"
#include "clang/ScalableStaticAnalysis/Core/Model/EntityLinkage.h"
#include "clang/ScalableStaticAnalysis/Core/Model/EntityName.h"
#include "clang/ScalableStaticAnalysis/Core/Serialization/JSONFormat.h"
#include "clang/ScalableStaticAnalysis/Core/TUSummary/ExtractorRegistry.h"
#include "clang/ScalableStaticAnalysis/Core/TUSummary/TUSummary.h"
#include "clang/ScalableStaticAnalysis/Core/TUSummary/TUSummaryBuilder.h"
#include "clang/ScalableStaticAnalysis/Core/WholeProgramAnalysis/AnalysisDriver.h"
#include "clang/ScalableStaticAnalysis/Core/WholeProgramAnalysis/WPASuite.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <vector>

using namespace clang;
using namespace ssaf;

namespace {

/// One VirtualMethodSummary, field for field, with entities spelled as letters.
/// By convention tests use uppercase letters for method entities and lowercase
/// ones for the slots they own, but the two are not distinguished: every letter
/// is just an entity.
struct MethodLayout {
  char Method;                 ///< Entity the summary is stored under.
  std::vector<char> Params;    ///< VirtualMethodSummary::ParamEntities.
  std::optional<char> Ret;     ///< VirtualMethodSummary::ReturnEntity.
  std::vector<char> Overrides; ///< VirtualMethodSummary::OverriddenMethods.
};

class UnsafeBufferReachableAnalysisTest : public TestFixture {
protected:
  using EPLEdge = std::pair<EntityPointerLevel, EntityPointerLevel>;

  static constexpr EntityLinkage ExternalLinkage =
      EntityLinkage(EntityLinkageType::External);

  std::unique_ptr<LUSummary> makeLUSummary() {
    NestedBuildNamespace NS(
        {BuildNamespace(BuildNamespaceKind::LinkUnit, "TestLU")});
    return std::make_unique<LUSummary>(llvm::Triple("arm64-apple-macosx"),
                                       std::move(NS));
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

  /// Insert a VirtualMethodSummary keyed by the method's own EntityId.
  void insertVirtualMethodSummary(LUSummary &LU, EntityId Id,
                                  VirtualMethodSummary Sum) {
    getData(LU)[VirtualMethodSummary::summaryName()][Id] =
        std::make_unique<VirtualMethodSummary>(std::move(Sum));
  }

  /// Insert a TypeConstrainedPointersEntitySummary for an entity.
  void insertTypeConstrainedPointersSummary(LUSummary &LU, EntityId Id,
                                            std::set<EntityId> Entities) {
    auto Sum = std::make_unique<TypeConstrainedPointersEntitySummary>();
    Sum->Entities = std::move(Entities);
    getData(LU)[TypeConstrainedPointersEntitySummary::summaryName()][Id] =
        std::move(Sum);
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
    return ROrErr->Reachables;
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

  /// Compute reachables for the virtual-method hierarchy described by
  /// \p Methods, seeded with \p StarterLayout, over the pointer-flow edges in
  /// \p EdgeLayout, and with the entities in \p Constrained being
  /// type-constrained. Starters, edges and type constraints all belong to one
  /// contributor, which no layout mentions.
  std::set<Node> familyClosure(llvm::ArrayRef<MethodLayout> Methods,
                               llvm::ArrayRef<Node> StarterLayout,
                               llvm::ArrayRef<Edge> EdgeLayout,
                               llvm::ArrayRef<char> Constrained,
                               unsigned Line) {
    constexpr char Contributor = '#';
    auto LU = makeLUSummary();
    auto Entities =
        createEntities(*LU, entityDomainOf(Contributor, Methods, StarterLayout,
                                           EdgeLayout, Constrained));
    auto GetEPL = [&Entities](const Node &N) -> EntityPointerLevel {
      return buildEntityPointerLevel(Entities[N.first], N.second);
    };

    auto GetIds = [&Entities](llvm::ArrayRef<char> Letters) {
      std::vector<EntityId> Ids;
      for (char L : Letters)
        Ids.push_back(Entities[L]);
      return Ids;
    };
    for (const MethodLayout &M : Methods) {
      VirtualMethodSummary Sum;
      Sum.ParamEntities = GetIds(M.Params);
      if (M.Ret)
        Sum.ReturnEntity = Entities[*M.Ret];
      Sum.OverriddenMethods = GetIds(M.Overrides);
      insertVirtualMethodSummary(*LU, Entities[M.Method], std::move(Sum));
    }

    std::vector<EPLEdge> Edges;
    for (const auto &[F, T] : EdgeLayout)
      Edges.push_back({GetEPL(F), GetEPL(T)});
    std::vector<EntityPointerLevel> Starters;
    for (const Node &N : StarterLayout)
      Starters.push_back(GetEPL(N));
    insertSummaries(*LU, Entities[Contributor], Edges, Starters);

    std::vector<EntityId> ConstrainedIds = GetIds(Constrained);
    insertTypeConstrainedPointersSummary(
        *LU, Entities[Contributor],
        {ConstrainedIds.begin(), ConstrainedIds.end()});

    auto Reachables = computeReachables(std::move(LU), Line);
    if (!Reachables)
      return {};

    std::set<Node> Result;
    for (const EntityPointerLevel &EPL : *Reachables)
      Result.insert({Entities[EPL.getEntity()], EPL.getPointerLevel()});
    return Result;
  }

  std::set<Node> familyClosure(llvm::ArrayRef<MethodLayout> Methods,
                               llvm::ArrayRef<Node> StarterLayout,
                               unsigned Line) {
    return familyClosure(Methods, StarterLayout, /*EdgeLayout=*/{},
                         /*Constrained=*/{}, Line);
  }

private:
  /// Every letter the layouts mention, plus \p Contributor, deduplicated.
  static std::vector<char> entityDomainOf(char Contributor,
                                          llvm::ArrayRef<MethodLayout> Methods,
                                          llvm::ArrayRef<Node> Starters,
                                          llvm::ArrayRef<Edge> Edges,
                                          llvm::ArrayRef<char> Constrained) {
    std::set<char> Domain{Contributor};
    for (const MethodLayout &M : Methods) {
      Domain.insert(M.Method);
      Domain.insert(M.Params.begin(), M.Params.end());
      Domain.insert(M.Overrides.begin(), M.Overrides.end());
      if (M.Ret)
        Domain.insert(*M.Ret);
    }
    for (const Node &N : Starters)
      Domain.insert(N.first);
    for (const auto &[From, To] : Edges) {
      Domain.insert(From.first);
      Domain.insert(To.first);
    }
    Domain.insert(Constrained.begin(), Constrained.end());
    return {Domain.begin(), Domain.end()};
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

// Multiple starters: (a,1) -> (b,1), (c,1) -> (d,1) (disconnected).
// Start from {(a,1), (c,1)} => {(a,1), (b,1), (c,1), (d,1)}
TEST_F(UnsafeBufferReachableAnalysisTest, MultipleStartersBothChains) {
  auto Reachables = singlePartition(
      /* EntityDomain */ {'a', 'b', 'c', 'd'},
      /* EdgeLayout */ {{{'a', 1}, {'b', 1}}, {{'c', 1}, {'d', 1}}},
      /* StarterLayout */ {{'a', 1}, {'c', 1}}, __LINE__);
  EXPECT_EQ(Reachables.size(), 4u);
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

// TODO: If one day we have good ways to query json in lit tests, move unit
// tests below to lit tests.

// Test harness for taking source code as input, driving all the separate tools
// (extractors and linking) up until UnsafeBufferReachableAnalysis.
class UnsafeBufferReachableAnalysisSourceTest : public TestFixture {
protected:
  using Node = std::pair<std::string, unsigned>;

  llvm::SmallString<128> TestDir;

  void SetUp() override {
    std::error_code EC = llvm::sys::fs::createUniqueDirectory(
        "unsafe-buffer-reachable-test", TestDir);
    ASSERT_FALSE(EC) << "Failed to create temp directory: " << EC.message();
  }

  void TearDown() override { llvm::sys::fs::remove_directories(TestDir); }

  llvm::SmallString<128> makePath(llvm::StringRef FileName) const {
    llvm::SmallString<128> Path = TestDir;
    llvm::sys::path::append(Path, FileName);
    return Path;
  }

  std::optional<std::set<Node>> computeReachables(llvm::StringRef Code,
                                                  unsigned Line) {
    std::unique_ptr<ASTUnit> AST = tooling::buildASTFromCodeWithArgs(
        Code, {"-Wno-unused-value", "-Wno-int-to-pointer-cast"});
    if (!AST) {
      ADD_FAILURE_AT(__FILE__, Line) << "failed to build AST";
      return std::nullopt;
    }

    SSAFOptions Opts;
    TUSummary TUSum(llvm::Triple("fake-unittest-triple"),
                    BuildNamespace(BuildNamespaceKind::CompilationUnit, "tu"));
    TUSummaryBuilder Builder(TUSum, Opts);

    for (llvm::StringRef ExtractorName :
         {PointerFlowEntitySummary::Name, UnsafeBufferUsageEntitySummary::Name,
          TypeConstrainedPointersEntitySummary::Name}) {
      std::unique_ptr<TUSummaryExtractor> Extractor =
          makeTUSummaryExtractor(ExtractorName, Builder);
      if (!Extractor) {
        ADD_FAILURE_AT(__FILE__, Line)
            << "failed to find extractor '" << ExtractorName << "'";
        return std::nullopt;
      }
      Extractor->HandleTranslationUnit(AST->getASTContext());
    }

    JSONFormat Format;
    llvm::SmallString<128> TUPath = makePath("tu.json");
    if (auto Err = Format.writeTUSummary(TUSum, TUPath)) {
      ADD_FAILURE_AT(__FILE__, Line) << llvm::toString(std::move(Err));
      return std::nullopt;
    }

    auto TUEncOrErr = Format.readTUSummaryEncoding(TUPath);
    if (!TUEncOrErr) {
      ADD_FAILURE_AT(__FILE__, Line) << llvm::toString(TUEncOrErr.takeError());
      return std::nullopt;
    }

    EntityLinker Linker(llvm::Triple("fake-unittest-triple"),
                        NestedBuildNamespace(BuildNamespace(
                            BuildNamespaceKind::LinkUnit, "lu")));
    if (auto Err = Linker.link(
            std::make_unique<TUSummaryEncoding>(std::move(*TUEncOrErr)))) {
      ADD_FAILURE_AT(__FILE__, Line) << llvm::toString(std::move(Err));
      return std::nullopt;
    }
    LUSummaryEncoding LUEnc = std::move(Linker).takeOutput();

    llvm::SmallString<128> LUPath = makePath("lu.json");
    if (auto Err = Format.writeLUSummaryEncoding(LUEnc, LUPath)) {
      ADD_FAILURE_AT(__FILE__, Line) << llvm::toString(std::move(Err));
      return std::nullopt;
    }

    // TearDown() removes the whole TestDir, but clean up these two
    // intermediate files as soon as we're done with them.
    auto Cleanup = llvm::scope_exit([&] {
      llvm::sys::fs::remove(TUPath);
      llvm::sys::fs::remove(LUPath);
    });

    auto LUOrErr = Format.readLUSummary(LUPath);
    if (!LUOrErr) {
      ADD_FAILURE_AT(__FILE__, Line) << llvm::toString(LUOrErr.takeError());
      return std::nullopt;
    }

    AnalysisDriver Driver(std::make_unique<LUSummary>(std::move(*LUOrErr)));
    auto WPAOrErr =
        Driver.run<PointerFlowAnalysisResult, UnsafeBufferUsageAnalysisResult,
                   TypeConstrainedPointersAnalysisResult,
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

    std::map<EntityId, std::string> IdToParamName;
    if (const FunctionDecl *FD = findFnByName("f", AST->getASTContext())) {
      for (const ParmVarDecl *PVD : FD->parameters()) {
        std::optional<EntityName> EN = getEntityName(PVD);
        if (!EN)
          continue;
        WPAOrErr->getIdTable().forEach(
            [&](const EntityName &Candidate, EntityId Id) {
              if (getSuffix(Candidate) == getSuffix(*EN))
                IdToParamName[Id] = PVD->getNameAsString();
            });
      }
    }

    std::set<Node> Result;
    for (const EntityPointerLevel &EPL : ROrErr->Reachables) {
      auto NameIt = IdToParamName.find(EPL.getEntity());
      if (NameIt == IdToParamName.end()) {
        ADD_FAILURE_AT(__FILE__, Line)
            << "reachable entity has no known source-level name";
        continue;
      }
      Result.insert({NameIt->second, EPL.getPointerLevel()});
    }
    return Result;
  }
};

// graph: (a,2)->(b,3)->(c,4)->(d,5)
TEST_F(UnsafeBufferReachableAnalysisSourceTest, LinearChain) {
  auto Reachables = computeReachables(R"cpp(
    void f(char **a, char ***b, char ****c, char *****d, int i) {
      a = *b;
      b = *c;
      c = *d;
      (*a)[i] = 0; // starter: (a,2)
    }
  )cpp",
                                      __LINE__);
  ASSERT_TRUE(Reachables);
  EXPECT_EQ(*Reachables,
            (std::set<Node>{{"a", 2}, {"b", 3}, {"c", 4}, {"d", 5}}));
}

// graph: (a,2)->(b,3); (b,4)->(c,1)->(d,1)
TEST_F(UnsafeBufferReachableAnalysisSourceTest, LinearChainDisconnected) {
  auto Reachables = computeReachables(R"cpp(
    void f(char ***a, char ****b, char *c, char *d, int i) {
      a = *b;
      ***b = c;
      c = d;
      (*a)[i] = 0; // starter: (a,2)
    }
  )cpp",
                                      __LINE__);
  ASSERT_TRUE(Reachables);
  EXPECT_EQ(*Reachables, (std::set<Node>{{"a", 2}, {"b", 3}}));
}

// graph: (a,2)->{(b,3),(c,3)}->(d,4)
TEST_F(UnsafeBufferReachableAnalysisSourceTest, Diamond) {
  auto Reachables = computeReachables(R"cpp(
    void f(char **a, char ***b, char ***c, char ****d, int i) {
      a = *b;
      a = *c;
      b = *d;
      c = *d;
      (*a)[i] = 0; // starter: (a,2)
    }
  )cpp",
                                      __LINE__);
  ASSERT_TRUE(Reachables);
  EXPECT_EQ(*Reachables,
            (std::set<Node>{{"a", 2}, {"b", 3}, {"c", 3}, {"d", 4}}));
}

// graph: (a,2)->{(b,3),(c,3)}; {(b,5),(c,5)}->(d,1)
TEST_F(UnsafeBufferReachableAnalysisSourceTest, DisconnectedDiamond) {
  auto Reachables = computeReachables(R"cpp(
    void f(char ****a, char *****b, char *****c, char *d, int i) {
      a = *b;
      a = *c;
      ****b = d;
      ****c = d;
      (*a)[i] = 0; // starter: (a,2)
    }
  )cpp",
                                      __LINE__);
  ASSERT_TRUE(Reachables);
  EXPECT_EQ(*Reachables, (std::set<Node>{{"a", 2}, {"b", 3}, {"c", 3}}));
}

// graph: (a,1)->(b,1); (c,1)->(d,1)
TEST_F(UnsafeBufferReachableAnalysisSourceTest, DisconnectedSubgraphs) {
  auto Reachables = computeReachables(R"cpp(
    void f(char *a, char *b, char *c, char *d, int i) {
      a = b;
      c = d;
      b[i] = 0; // starter: (b,1)
    }
  )cpp",
                                      __LINE__);
  ASSERT_TRUE(Reachables);
  EXPECT_EQ(*Reachables, (std::set<Node>{{"b", 1}}));
}

// graph: (a,2)->(b,2)->(c,2)->(d,2)->(a,2)
TEST_F(UnsafeBufferReachableAnalysisSourceTest, Cycle) {
  auto Reachables = computeReachables(R"cpp(
    void f(char **a, char **b, char **c, char **d, int i) {
      a = b;
      b = c;
      c = d;
      d = a;
      (*c)[i] = 0; // starter: (c,2)
    }
  )cpp",
                                      __LINE__);
  ASSERT_TRUE(Reachables);
  EXPECT_EQ(*Reachables,
            (std::set<Node>{{"a", 2}, {"b", 2}, {"c", 2}, {"d", 2}}));
}

// graph: (a,2)->(b,3)->(c,4)->(d,5)->(a,1)
TEST_F(UnsafeBufferReachableAnalysisSourceTest, CycleIncreasing) {
  auto Reachables = computeReachables(R"cpp(
    void f(char **a, char ***b, char ****c, char *****d, int i) {
      a = *b;
      *b = **c;
      **c = ***d;
      ***d = a;
      (*a)[i] = 0; // starter: (a,2)
    }
  )cpp",
                                      __LINE__);
  ASSERT_TRUE(Reachables);
  EXPECT_EQ(*Reachables,
            (std::set<Node>{{"a", 2}, {"b", 3}, {"c", 4}, {"d", 5}}));
}

// graph: (a,2)->{(b,3),(c,3),(d,3)}
TEST_F(UnsafeBufferReachableAnalysisSourceTest, StarFromHub) {
  auto Reachables = computeReachables(R"cpp(
    void f(char **a, char ***b, char ***c, char ***d, int i) {
      a = *b;
      a = *c;
      a = *d;
      (*a)[i] = 0; // starter: (a,2)
    }
  )cpp",
                                      __LINE__);
  ASSERT_TRUE(Reachables);
  EXPECT_EQ(*Reachables,
            (std::set<Node>{{"a", 2}, {"b", 3}, {"c", 3}, {"d", 3}}));
}

// graph: (a,2)->{(b,1),(c,1),(d,1)}
TEST_F(UnsafeBufferReachableAnalysisSourceTest, StarFromHubBelowEdge) {
  auto Reachables = computeReachables(R"cpp(
    void f(char **a, char *b, char *c, char *d, int i) {
      *a = b;
      *a = c;
      *a = d;
      a[i] = 0; // starter: (a,1)
    }
  )cpp",
                                      __LINE__);
  ASSERT_TRUE(Reachables);
  EXPECT_EQ(*Reachables, (std::set<Node>{{"a", 1}}));
}

// graph: {(a,2),(b,2),(c,2)}->(d,3)
TEST_F(UnsafeBufferReachableAnalysisSourceTest, ReverseStarFromSource) {
  auto Reachables = computeReachables(R"cpp(
    void f(char **a, char **b, char **c, char ***d, int i) {
      a = *d;
      b = *d;
      c = *d;
      (*a)[i] = 0; // starter: (a,2)
    }
  )cpp",
                                      __LINE__);
  ASSERT_TRUE(Reachables);
  EXPECT_EQ(*Reachables, (std::set<Node>{{"a", 2}, {"d", 3}}));
}

// graph: {(a,2),(b,2),(c,2)}->(d,2)
TEST_F(UnsafeBufferReachableAnalysisSourceTest, ReverseStarFromSink) {
  auto Reachables = computeReachables(R"cpp(
    void f(char **a, char **b, char **c, char **d, int i) {
      a = d;
      b = d;
      c = d;
      (*d)[i] = 0; // starter: (d,2)
    }
  )cpp",
                                      __LINE__);
  ASSERT_TRUE(Reachables);
  EXPECT_EQ(*Reachables, (std::set<Node>{{"d", 2}}));
}

// graph: (a,2)->(b,2)->(b,2)->(c,3)->(d,4)
TEST_F(UnsafeBufferReachableAnalysisSourceTest, SelfLoopFromRoot) {
  auto Reachables = computeReachables(R"cpp(
    void f(char **a, char **b, char ***c, char ****d, int i) {
      a = b;
      b = b;
      b = *c;
      c = *d;
      (*a)[i] = 0; // starter: (a,2)
    }
  )cpp",
                                      __LINE__);
  ASSERT_TRUE(Reachables);
  EXPECT_EQ(*Reachables,
            (std::set<Node>{{"a", 2}, {"b", 2}, {"c", 3}, {"d", 4}}));
}

// graph: (b,2)->(b,2)->(c,3)->(d,4)
TEST_F(UnsafeBufferReachableAnalysisSourceTest, SelfLoopFromLoopNode) {
  auto Reachables = computeReachables(R"cpp(
    void f(char **a, char **b, char ***c, char ****d, int i) {
      a = b;
      b = b;
      b = *c;
      c = *d;
      (*b)[i] = 0; // starter: (b,2)
    }
  )cpp",
                                      __LINE__);
  ASSERT_TRUE(Reachables);
  EXPECT_EQ(*Reachables, (std::set<Node>{{"b", 2}, {"c", 3}, {"d", 4}}));
}

// graph: (a,2)->(b,3); (c,2)->(d,3)
TEST_F(UnsafeBufferReachableAnalysisSourceTest, MultipleStarters) {
  auto Reachables = computeReachables(R"cpp(
    void f(char **a, char ***b, char **c, char ***d, int i) {
      a = *b;
      c = *d;
      (*a)[i] = 0; // starter: (a,2)
      (*c)[i] = 0; // starter: (c,2)
    }
  )cpp",
                                      __LINE__);
  ASSERT_TRUE(Reachables);
  EXPECT_EQ(*Reachables,
            (std::set<Node>{{"a", 2}, {"b", 3}, {"c", 2}, {"d", 3}}));
}

// graph: (a,3)->{(b,3),(c,2)}
TEST_F(UnsafeBufferReachableAnalysisSourceTest, MultipleKeysSameEntity) {
  auto Reachables = computeReachables(R"cpp(
    void f(char ***a, char ***b, char **c, int i) {
      a = b;
      *a = c;
      (**a)[i] = 0; // starter: (a,3)
    }
  )cpp",
                                      __LINE__);
  ASSERT_TRUE(Reachables);
  EXPECT_EQ(*Reachables, (std::set<Node>{{"a", 3}, {"b", 3}, {"c", 2}}));
}

////////////////////////////////////////////////////////////////////////////////
// Family-closure tests
////////////////////////////////////////////////////////////////////////////////

// Method B owns param slot p; D overrides B and owns param slot q.
// Seeding (q,1) mirrors (p,1).
TEST_F(UnsafeBufferReachableAnalysisTest, FamilyClosureParamSlot) {
  auto Reachables = familyClosure(
      /* Methods */ {{'B', /*Params=*/{'p'}, /*Ret=*/{}, /*Overrides=*/{}},
                     {'D', /*Params=*/{'q'}, /*Ret=*/{}, /*Overrides=*/{'B'}}},
      /* Starters */ {{'q', 1}}, __LINE__);

  EXPECT_EQ(Reachables, (std::set<Node>{
                            {'q', 1},
                            {'p', 1}, // Up to the overridden method.
                        }));
}

// As above, but p and q are the methods' return slots rather than parameters.
TEST_F(UnsafeBufferReachableAnalysisTest, FamilyClosureReturnSlot) {
  auto Reachables = familyClosure(
      /* Methods */ {{'B', /*Params=*/{}, /*Ret=*/{'p'}, /*Overrides=*/{}},
                     {'D', /*Params=*/{}, /*Ret=*/{'q'}, /*Overrides=*/{'B'}}},
      /* Starters */ {{'q', 1}}, __LINE__);

  EXPECT_EQ(Reachables, (std::set<Node>{
                            {'q', 1},
                            {'p', 1}, // Up to the overridden method.
                        }));
}

// No virtual methods at all, so no families. Closure adds nothing, so the
// starters are all that is reachable.
TEST_F(UnsafeBufferReachableAnalysisTest, FamilyClosureEmptyFamilyIsNoop) {
  auto Reachables =
      familyClosure(/* Methods */ {}, /* Starters */ {{'b', 1}}, __LINE__);

  EXPECT_EQ(Reachables, (std::set<Node>{{'b', 1}}));
}

// X and Y both override B, so all three slots share one family.
// Seeding X propagates up to B *and* sideways to the sibling override Y.
TEST_F(UnsafeBufferReachableAnalysisTest, FamilyClosureThreeMemberFamily) {
  auto Reachables = familyClosure(
      /* Methods */ {{'B', /*Params=*/{'p'}, /*Ret=*/{}, /*Overrides=*/{}},
                     {'X', /*Params=*/{'x'}, /*Ret=*/{}, /*Overrides=*/{'B'}},
                     {'Y', /*Params=*/{'y'}, /*Ret=*/{}, /*Overrides=*/{'B'}}},
      /* Starters */ {{'x', 1}}, __LINE__);

  EXPECT_EQ(Reachables, (std::set<Node>{
                            {'x', 1},
                            {'p', 1}, // Up to the base.
                            {'y', 1}, // Sideways to the sibling override.
                        }));
}

// Family closure is level-preserving: an EPL reachable at level 3 propagates to
// the family member at level 3 only, not to the levels below it.
TEST_F(UnsafeBufferReachableAnalysisTest, FamilyClosurePreservesPointerLevel) {
  auto Reachables = familyClosure(
      /* Methods */ {{'B', /*Params=*/{'p'}, /*Ret=*/{}, /*Overrides=*/{}},
                     {'D', /*Params=*/{'q'}, /*Ret=*/{}, /*Overrides=*/{'B'}}},
      /* Starters */ {{'q', 3}}, __LINE__);

  EXPECT_EQ(Reachables, (std::set<Node>{
                            {'q', 3},
                            {'p', 3}, // Level 3 only; neither p@1 nor p@2.
                        }));
}

// A single slot reachable at several levels propagates every one of those
// levels onto its family members.
TEST_F(UnsafeBufferReachableAnalysisTest, FamilyClosureMultipleLevelsSameSlot) {
  auto Reachables = familyClosure(
      /* Methods */ {{'B', /*Params=*/{'p'}, /*Ret=*/{}, /*Overrides=*/{}},
                     {'D', /*Params=*/{'q'}, /*Ret=*/{}, /*Overrides=*/{'B'}}},
      /* Starters */ {{'q', 1}, {'q', 2}}, __LINE__);

  EXPECT_EQ(Reachables, (std::set<Node>{
                            {'q', 1},
                            {'q', 2},
                            {'p', 1}, // Both levels, not just one of them.
                            {'p', 2},
                        }));
}

// (p,1) becomes reachable only via family closure, and (p,1) -> (z,1) is a
// flow edge. Nothing is seeded at (p,1), so only the family closure can make
// the pointer-flow search visit it.
TEST_F(UnsafeBufferReachableAnalysisTest, FamilyClosureFeedsBackIntoDFS) {
  auto Reachables = familyClosure(
      /* Methods */ {{'B', /*Params=*/{'p'}, /*Ret=*/{}, /*Overrides=*/{}},
                     {'D', /*Params=*/{'q'}, /*Ret=*/{}, /*Overrides=*/{'B'}}},
      /* Starters */ {{'q', 1}},
      /* EdgeLayout */ {{{'p', 1}, {'z', 1}}},
      /* Constrained */ {}, __LINE__);

  EXPECT_EQ(Reachables, (std::set<Node>{
                            {'q', 1},
                            {'p', 1}, // Up to the base.
                            {'z', 1}, // Flow successor of the mirrored (p,1).
                        }));
}

// X and Y both override B, whose slot p is type-constrained. Seeding X reaches
// the sibling Y through the family, but never the constrained p (C3).
TEST_F(UnsafeBufferReachableAnalysisTest,
       FamilyClosureSkipsTypeConstrainedMember) {
  auto Reachables = familyClosure(
      /* Methods */ {{'B', /*Params=*/{'p'}, /*Ret=*/{}, /*Overrides=*/{}},
                     {'X', /*Params=*/{'x'}, /*Ret=*/{}, /*Overrides=*/{'B'}},
                     {'Y', /*Params=*/{'y'}, /*Ret=*/{}, /*Overrides=*/{'B'}}},
      /* Starters */ {{'x', 1}},
      /* EdgeLayout */ {},
      /* Constrained */ {'p'}, __LINE__);

  EXPECT_EQ(Reachables, (std::set<Node>{
                            {'x', 1},
                            {'y', 1}, // Sideways, even though p is excluded.
                        }));
}

} // namespace
