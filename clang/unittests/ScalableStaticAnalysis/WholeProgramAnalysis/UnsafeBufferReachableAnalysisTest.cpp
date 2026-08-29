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
    for (const auto &[Id, EPLs] : ROrErr->Reachables) {
      for (const EntityPointerLevel &EPL : EPLs) {
        auto NameIt = IdToParamName.find(EPL.getEntity());
        if (NameIt == IdToParamName.end()) {
          ADD_FAILURE_AT(__FILE__, Line)
              << "reachable entity has no known source-level name";
          continue;
        }
        Result.insert({NameIt->second, EPL.getPointerLevel()});
      }
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

} // namespace
