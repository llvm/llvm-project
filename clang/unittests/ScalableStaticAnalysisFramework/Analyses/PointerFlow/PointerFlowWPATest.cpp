//===- PointerFlowWPATest.cpp ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TestFixture.h"
#include "clang/ScalableStaticAnalysisFramework/Analyses/EntityPointerLevel/EntityPointerLevel.h"
#include "clang/ScalableStaticAnalysisFramework/Analyses/PointerFlow/PointerFlow.h"
#include "clang/ScalableStaticAnalysisFramework/Analyses/PointerFlow/PointerFlowAnalysis.h"
#include "clang/ScalableStaticAnalysisFramework/Core/EntityLinker/LUSummary.h"
#include "clang/ScalableStaticAnalysisFramework/Core/Model/BuildNamespace.h"
#include "clang/ScalableStaticAnalysisFramework/Core/Model/EntityId.h"
#include "clang/ScalableStaticAnalysisFramework/Core/Model/EntityLinkage.h"
#include "clang/ScalableStaticAnalysisFramework/Core/Model/EntityName.h"
#include "clang/ScalableStaticAnalysisFramework/Core/WholeProgramAnalysis/AnalysisDriver.h"
#include "clang/ScalableStaticAnalysisFramework/Core/WholeProgramAnalysis/WPASuite.h"
#include "llvm/Testing/Support/Error.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <memory>

using namespace clang;
using namespace ssaf;
using testing::IsEmpty;
using testing::UnorderedElementsAre;

PointerFlowEntitySummary ssaf::buildPointerFlowEntitySummary(EdgeSet Edges);

namespace {

class PointerFlowWPATest : public TestFixture {
protected:
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

  void insertSummary(LUSummary &LU, EntityId Id, EdgeSet Edges) {
    getData(LU)[PointerFlowEntitySummary::summaryName()][Id] =
        std::make_unique<PointerFlowEntitySummary>(
            buildPointerFlowEntitySummary(std::move(Edges)));
  }

  const PointerFlowAnalysisResult &
  runAnalysisForResult(std::unique_ptr<LUSummary> LU) {
    AnalysisDriver Driver(std::move(LU));
    auto WPAOrErr = Driver.run<PointerFlowAnalysisResult>();
    EXPECT_THAT_EXPECTED(WPAOrErr, llvm::Succeeded());
    WPA = std::move(*WPAOrErr);
    auto ROrErr = WPA.get<PointerFlowAnalysisResult>();
    EXPECT_THAT_EXPECTED(ROrErr, llvm::Succeeded());
    return *ROrErr;
  }

private:
  WPASuite WPA = makeWPASuite();
};

TEST_F(PointerFlowWPATest, SingleEntitySummary) {
  auto LU = makeLUSummary();
  EntityId Foo = addEntity(*LU, "c:@F@foo");
  EntityId Src = addEntity(*LU, "c:@F@foo@src");
  EntityId Dest = addEntity(*LU, "c:@F@foo@dest");
  auto Src1 = buildEntityPointerLevel(Src, 1);
  auto Dest1 = buildEntityPointerLevel(Dest, 1);
  auto Dest2 = buildEntityPointerLevel(Dest, 2);
  EdgeSet Edges;

  Edges[Src1] = {Dest1, Dest2};
  insertSummary(*LU, Foo, std::move(Edges));

  const auto &Result = runAnalysisForResult(std::move(LU));

  ASSERT_EQ(Result.Edges.size(), 1u);
  EXPECT_EQ(Result.Edges.at(Foo).size(), 1u);
  EXPECT_THAT(Result.Edges.at(Foo).at(Src1),
              UnorderedElementsAre(Dest1, Dest2));
}

TEST_F(PointerFlowWPATest, MultiEntitySummaries) {
  auto LU = makeLUSummary();
  EntityId Foo = addEntity(*LU, "c:@F@foo");
  EntityId Bar = addEntity(*LU, "c:@F@bar");
  EntityId FooSrc = addEntity(*LU, "c:@F@foo@src");
  EntityId FooDest = addEntity(*LU, "c:@F@foo@dest");
  EntityId BarSrc = addEntity(*LU, "c:@F@bar@src");
  EntityId BarDest = addEntity(*LU, "c:@F@bar@dest");
  auto FooSrc1 = buildEntityPointerLevel(FooSrc, 1);
  auto BarSrc1 = buildEntityPointerLevel(BarSrc, 1);
  EntityPointerLevelSet FooDests, BarDests;

  for (unsigned I = 0; I < 5; ++I) {
    FooDests.insert(buildEntityPointerLevel(FooDest, I));
    BarDests.insert(buildEntityPointerLevel(BarDest, I));
  }

  EdgeSet FooEdges;
  FooEdges[FooSrc1] = FooDests;
  insertSummary(*LU, Foo, std::move(FooEdges));

  EdgeSet BarEdges;
  BarEdges[BarSrc1] = BarDests;
  insertSummary(*LU, Bar, std::move(BarEdges));

  const auto &R = runAnalysisForResult(std::move(LU));

  EXPECT_EQ(R.Edges.size(), 2u);
  EXPECT_EQ(R.Edges.at(Foo).at(FooSrc1), FooDests);
  EXPECT_EQ(R.Edges.at(Bar).at(BarSrc1), BarDests);
}

TEST_F(PointerFlowWPATest, EmptySummary) {
  auto LU = makeLUSummary();
  EntityId Foo = addEntity(*LU, "c:@F@foo");

  insertSummary(*LU, Foo, {});

  const auto &R = runAnalysisForResult(std::move(LU));

  ASSERT_EQ(R.Edges.count(Foo), 1u);
  EXPECT_THAT(R.Edges.at(Foo), IsEmpty());
}

} // namespace
