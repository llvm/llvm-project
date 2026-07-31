//===- UnsafeBufferReachableAnalysisTest.cpp ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../TestFixture.h"
#include "clang/ScalableStaticAnalysis/Analyses/EntityPointerLevel/EntityPointerLevel.h"
#include "clang/ScalableStaticAnalysis/Analyses/PointerFlow/PointerFlow.h"
#include "clang/ScalableStaticAnalysis/Analyses/PointerFlow/PointerFlowAnalysis.h"
#include "clang/ScalableStaticAnalysis/Analyses/UnsafeBufferUsage/UnsafeBufferUsage.h"
#include "clang/ScalableStaticAnalysis/Analyses/UnsafeBufferUsage/UnsafeBufferUsageAnalysis.h"
#include "clang/ScalableStaticAnalysis/Analyses/VirtualMethodFamily/VirtualMethodFamily.h"
#include "clang/ScalableStaticAnalysis/Core/EntityLinker/LUSummary.h"
#include "clang/ScalableStaticAnalysis/Core/Model/BuildNamespace.h"
#include "clang/ScalableStaticAnalysis/Core/Model/EntityId.h"
#include "clang/ScalableStaticAnalysis/Core/Model/EntityLinkage.h"
#include "clang/ScalableStaticAnalysis/Core/Model/EntityName.h"
#include "clang/ScalableStaticAnalysis/Core/WholeProgramAnalysis/AnalysisDriver.h"
#include "clang/ScalableStaticAnalysis/Core/WholeProgramAnalysis/WPASuite.h"
#include "llvm/ADT/ArrayRef.h"
#include "gtest/gtest.h"
#include <map>
#include <memory>
#include <optional>
#include <ostream>
#include <set>
#include <utility>
#include <vector>

using namespace clang;
using namespace ssaf;

namespace clang::ssaf {
extern PointerFlowEntitySummary buildPointerFlowEntitySummary(EdgeSet Edges);
extern UnsafeBufferUsageEntitySummary
    buildUnsafeBufferUsageEntitySummary(EntityPointerLevelSet);
} // namespace clang::ssaf

namespace {

/// An entity at a pointer level, with the entity spelled as the letter naming
/// it in a test's layout.
using Node = std::pair<char, unsigned>;
using Edge = std::pair<Node, Node>;

/// One VirtualMethodSummary, field for field, with entities spelled as letters.
/// By convention tests use uppercase letters for method entities and lowercase
/// ones for the slots they own, but the two are not distinguished: every letter
/// is just an entity.
struct MethodLayout {
  char Method;                 ///< Scope the summary is stored under.
  std::vector<char> Params;    ///< VirtualMethodSummary::ParamEntities.
  std::optional<char> Ret;     ///< VirtualMethodSummary::ReturnEntity.
  std::vector<char> Overrides; ///< VirtualMethodSummary::OverriddenMethods.
};

/// An edge, and the scope (method entity) whose pointer-flow graph owns it.
using ScopedEdge = std::pair<char, Edge>;

/// Nodes grouped by the scope (method entity) that owns them: the starter
/// layout of a family-closure test, the entries its closure pass is expected to
/// add, and the `Reachables` map it produces are all spelled this way.
///
/// Wrapped in a class rather than used as a plain map so that it can be added
/// to with `operator+` and so that a failing comparison prints
/// "{B: p@1, p@2; D: q@1}" instead of gtest's raw char dump.
class ScopedNodes {
  std::map<char, std::set<Node>> Scopes;

public:
  using value_type = std::map<char, std::set<Node>>::value_type;

  ScopedNodes() = default;
  ScopedNodes(std::initializer_list<value_type> Init) : Scopes(Init) {}

  void insert(char Scope, Node N) { Scopes[Scope].insert(N); }

  const std::map<char, std::set<Node>> &scopes() const { return Scopes; }

  bool operator==(const ScopedNodes &Other) const {
    return Scopes == Other.Scopes;
  }
  bool operator!=(const ScopedNodes &Other) const { return !(*this == Other); }
};

/// The union of \p L and \p R. Lets a test state its expectation as "the
/// starters, plus what closure adds".
ScopedNodes operator+(const ScopedNodes &L, const ScopedNodes &R) {
  ScopedNodes Result = L;
  for (const auto &[Scope, Nodes] : R.scopes())
    for (const Node &N : Nodes)
      Result.insert(Scope, N);
  return Result;
}

void PrintTo(const ScopedNodes &SN, std::ostream *OS) {
  *OS << "{";
  const char *ScopeSep = "";
  for (const auto &[Scope, Nodes] : SN.scopes()) {
    *OS << ScopeSep << Scope << ": ";
    ScopeSep = "; ";
    const char *NodeSep = "";
    for (const auto &[Slot, Level] : Nodes) {
      *OS << NodeSep << Slot << "@" << Level;
      NodeSep = ", ";
    }
  }
  *OS << "}";
}

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

  /// Insert a VirtualMethodSummary keyed by the method's own EntityId.
  void insertVirtualMethodSummary(LUSummary &LU, EntityId Id,
                                  VirtualMethodSummary Sum) {
    getData(LU)[VirtualMethodSummary::summaryName()][Id] =
        std::make_unique<VirtualMethodSummary>(std::move(Sum));
  }

  /// Build a VirtualMethodSummary from its parameter entities, optional
  /// return-slot entity, and direct override edges.
  VirtualMethodSummary makeMethodSummary(std::vector<EntityId> ParamEntities,
                                         std::optional<EntityId> RetEntity,
                                         std::vector<EntityId> Overridden) {
    VirtualMethodSummary S;
    S.ParamEntities = std::move(ParamEntities);
    S.ReturnEntity = RetEntity;
    S.OverriddenMethods = std::move(Overridden);
    return S;
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

  /// Run the driver and return the full per-scope `Reachables` map.
  std::optional<std::map<EntityId, EntityPointerLevelSet>>
  computeReachablesByScope(std::unique_ptr<LUSummary> LU, unsigned Line) {
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

  /// Run the driver and return the reachable EPLs of every scope, flattened
  /// into a single set.
  std::optional<EntityPointerLevelSet>
  computeReachables(std::unique_ptr<LUSummary> LU, unsigned Line) {
    auto ByScope = computeReachablesByScope(std::move(LU), Line);
    if (!ByScope)
      return std::nullopt;
    EntityPointerLevelSet Result;
    for (const EntityPointerLevelSet &EPLs : llvm::make_second_range(*ByScope))
      Result.insert(EPLs.begin(), EPLs.end());
    return Result;
  }

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

  /// Compute reachables per scope for the virtual-method hierarchy described by
  /// \p Methods, seeded with \p StarterLayout and, optionally, the pointer-flow
  /// edges in \p EdgeLayout. Both starters and edges name the scope that owns
  /// them, because family closure is about which scope an EPL ends up under.
  ///
  /// The entity domain is the set of letters the layouts mention. Only scopes
  /// named by a starter or an edge get PointerFlow/UnsafeBufferUsage summaries:
  /// UnsafeBufferUsageAnalysis records even an empty summary, which would then
  /// show up as an empty entry in `Reachables`.
  ScopedNodes familyClosure(llvm::ArrayRef<MethodLayout> Methods,
                            const ScopedNodes &StarterLayout,
                            llvm::ArrayRef<ScopedEdge> EdgeLayout,
                            unsigned Line) {
    auto LU = makeLUSummary();
    auto Entities =
        createEntities(*LU, entityDomainOf(Methods, StarterLayout, EdgeLayout));
    auto GetEPL = [&Entities](const Node &N) -> EntityPointerLevel {
      return buildEntityPointerLevel(Entities[N.first], N.second);
    };

    for (const MethodLayout &M : Methods) {
      std::vector<EntityId> Params;
      for (char P : M.Params)
        Params.push_back(Entities[P]);
      std::vector<EntityId> Overrides;
      for (char O : M.Overrides)
        Overrides.push_back(Entities[O]);
      std::optional<EntityId> Ret;
      if (M.Ret)
        Ret = Entities[*M.Ret];
      insertVirtualMethodSummary(
          *LU, Entities[M.Method],
          makeMethodSummary(std::move(Params), Ret, std::move(Overrides)));
    }

    std::map<char, std::vector<EPLEdge>> EdgesOfScope;
    std::map<char, std::vector<EntityPointerLevel>> StartersOfScope;
    for (const auto &[Scope, E] : EdgeLayout)
      EdgesOfScope[Scope].push_back({GetEPL(E.first), GetEPL(E.second)});
    for (const auto &[Scope, Nodes] : StarterLayout.scopes())
      for (const Node &N : Nodes)
        StartersOfScope[Scope].push_back(GetEPL(N));

    std::set<char> Scopes;
    for (char Scope : llvm::make_first_range(EdgesOfScope))
      Scopes.insert(Scope);
    for (char Scope : llvm::make_first_range(StartersOfScope))
      Scopes.insert(Scope);
    for (char Scope : Scopes)
      insertSummaries(*LU, Entities[Scope], EdgesOfScope[Scope],
                      StartersOfScope[Scope]);

    auto Reachables = computeReachablesByScope(std::move(LU), Line);
    if (!Reachables)
      return {};

    ScopedNodes Result;
    for (const auto &[Scope, EPLs] : *Reachables)
      for (const EntityPointerLevel &EPL : EPLs)
        Result.insert(Entities[Scope],
                      {Entities[EPL.getEntity()], EPL.getPointerLevel()});
    return Result;
  }

  ScopedNodes familyClosure(llvm::ArrayRef<MethodLayout> Methods,
                            const ScopedNodes &StarterLayout, unsigned Line) {
    return familyClosure(Methods, StarterLayout, /*EdgeLayout=*/{}, Line);
  }

private:
  /// Every letter the layouts mention, deduplicated.
  static std::vector<char> entityDomainOf(llvm::ArrayRef<MethodLayout> Methods,
                                          const ScopedNodes &Starters,
                                          llvm::ArrayRef<ScopedEdge> Edges) {
    std::set<char> Domain;
    for (const MethodLayout &M : Methods) {
      Domain.insert(M.Method);
      Domain.insert(M.Params.begin(), M.Params.end());
      Domain.insert(M.Overrides.begin(), M.Overrides.end());
      if (M.Ret)
        Domain.insert(*M.Ret);
    }
    for (const auto &[Scope, Nodes] : Starters.scopes()) {
      Domain.insert(Scope);
      for (const Node &N : Nodes)
        Domain.insert(N.first);
    }
    for (const auto &[Scope, E] : Edges) {
      Domain.insert(Scope);
      Domain.insert(E.first.first);
      Domain.insert(E.second.first);
    }
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

////////////////////////////////////////////////////////////////////////////////
// Family-closure tests
////////////////////////////////////////////////////////////////////////////////

// Method B owns param slot p; D overrides B and owns param slot q.
// Seeding (q,1) in D's scope mirrors (p,1) into B's scope.
TEST_F(UnsafeBufferReachableAnalysisTest, FamilyClosureParamSlot) {
  const ScopedNodes Starters{{'D', {{'q', 1}}}};
  auto Reachables = familyClosure(
      /* Methods */ {{'B', /*Params=*/{'p'}, /*Ret=*/{}, /*Overrides=*/{}},
                     {'D', /*Params=*/{'q'}, /*Ret=*/{}, /*Overrides=*/{'B'}}},
      Starters, __LINE__);

  const ScopedNodes AddedByClosure{
      {'B', {{'p', 1}}}, // Up to the overridden method.
  };
  EXPECT_EQ(Reachables, Starters + AddedByClosure);
}

// As above, but p and q are the methods' return slots rather than parameters.
TEST_F(UnsafeBufferReachableAnalysisTest, FamilyClosureReturnSlot) {
  const ScopedNodes Starters{{'D', {{'q', 1}}}};
  auto Reachables = familyClosure(
      /* Methods */ {{'B', /*Params=*/{}, /*Ret=*/{'p'}, /*Overrides=*/{}},
                     {'D', /*Params=*/{}, /*Ret=*/{'q'}, /*Overrides=*/{'B'}}},
      Starters, __LINE__);

  const ScopedNodes AddedByClosure{
      {'B', {{'p', 1}}}, // Up to the overridden method.
  };
  EXPECT_EQ(Reachables, Starters + AddedByClosure);
}

// No virtual methods at all, so no families: f is an ordinary function and b a
// buffer it owns. Closure adds nothing, so the starters are all that is
// reachable.
TEST_F(UnsafeBufferReachableAnalysisTest, FamilyClosureEmptyFamilyIsNoop) {
  const ScopedNodes Starters{{'f', {{'b', 1}}}};
  auto Reachables = familyClosure(/* Methods */ {}, Starters, __LINE__);

  EXPECT_EQ(Reachables, Starters + /*AddedByClosure=*/ScopedNodes{});
}

// X and Y both override B, so all three slots share one family.
// Seeding X propagates up to B *and* sideways to the sibling override Y.
TEST_F(UnsafeBufferReachableAnalysisTest, FamilyClosureThreeMemberFamily) {
  const ScopedNodes Starters{{'X', {{'x', 1}}}};
  auto Reachables = familyClosure(
      /* Methods */ {{'B', /*Params=*/{'p'}, /*Ret=*/{}, /*Overrides=*/{}},
                     {'X', /*Params=*/{'x'}, /*Ret=*/{}, /*Overrides=*/{'B'}},
                     {'Y', /*Params=*/{'y'}, /*Ret=*/{}, /*Overrides=*/{'B'}}},
      Starters, __LINE__);

  const ScopedNodes AddedByClosure{
      {'B', {{'p', 1}}}, // Up to the base.
      {'Y', {{'y', 1}}}, // Sideways to the sibling override.
  };
  EXPECT_EQ(Reachables, Starters + AddedByClosure);
}

// Family closure is level-preserving: an EPL reachable at level 3 propagates to
// the family member at level 3 only, not to the levels below it.
TEST_F(UnsafeBufferReachableAnalysisTest, FamilyClosurePreservesPointerLevel) {
  const ScopedNodes Starters{{'D', {{'q', 3}}}};
  auto Reachables = familyClosure(
      /* Methods */ {{'B', /*Params=*/{'p'}, /*Ret=*/{}, /*Overrides=*/{}},
                     {'D', /*Params=*/{'q'}, /*Ret=*/{}, /*Overrides=*/{'B'}}},
      Starters, __LINE__);

  const ScopedNodes AddedByClosure{
      {'B', {{'p', 3}}}, // Level 3 only; neither p@1 nor p@2.
  };
  EXPECT_EQ(Reachables, Starters + AddedByClosure);
}

// A single slot reachable at several levels propagates every one of those
// levels onto its family members.
TEST_F(UnsafeBufferReachableAnalysisTest, FamilyClosureMultipleLevelsSameSlot) {
  const ScopedNodes Starters{{'D', {{'q', 1}, {'q', 2}}}};
  auto Reachables = familyClosure(
      /* Methods */ {{'B', /*Params=*/{'p'}, /*Ret=*/{}, /*Overrides=*/{}},
                     {'D', /*Params=*/{'q'}, /*Ret=*/{}, /*Overrides=*/{'B'}}},
      Starters, __LINE__);

  const ScopedNodes AddedByClosure{
      {'B', {{'p', 1}, {'p', 2}}}, // Both levels, not just one of them.
  };
  EXPECT_EQ(Reachables, Starters + AddedByClosure);
}

// Pins the known limitation documented by the FIXME on `runFamilyClosurePass`:
// the closure pass runs *after* the pointer-flow DFS has converged, and
// `step()` returns false unconditionally, so EPLs discovered by the closure are
// never fed back through the pointer-flow graph.
//
// Here (p,1) becomes reachable only via family closure, and B owns the flow
// edge (p,1) -> (z,1). Nothing is seeded in B's scope, so the DFS never visits
// (p,1) on its own. A true fixpoint over DFS + closure would also reach (z,1).
TEST_F(UnsafeBufferReachableAnalysisTest, FamilyClosureDoesNotRerunDFS) {
  const ScopedNodes Starters{{'D', {{'q', 1}}}};
  auto Reachables = familyClosure(
      /* Methods */ {{'B', /*Params=*/{'p'}, /*Ret=*/{}, /*Overrides=*/{}},
                     {'D', /*Params=*/{'q'}, /*Ret=*/{}, /*Overrides=*/{'B'}}},
      Starters,
      /* EdgeLayout */ {{'B', {{'p', 1}, {'z', 1}}}}, __LINE__);

  // FIXME: z@1 belongs in B's set below -- it is a pointer-flow successor of
  // (p,1), which closure just discovered. It is missed because the pass does
  // not re-run the DFS over its own output.
  const ScopedNodes AddedByClosure{
      {'B', {{'p', 1}}}, // Up to the base, but no z@1.
  };
  EXPECT_EQ(Reachables, Starters + AddedByClosure)
      << "If z@1 shows up in B's scope, the DFS/closure fixpoint gap was "
         "fixed; add it to AddedByClosure above";
}

} // namespace
