//===- UnsafeBufferUsageWPATest.cpp ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TestFixture.h"
#include "clang/ScalableStaticAnalysisFramework/Analyses/EntityPointerLevel/EntityPointerLevel.h"
#include "clang/ScalableStaticAnalysisFramework/Analyses/UnsafeBufferUsage/UnsafeBufferUsage.h"
#include "clang/ScalableStaticAnalysisFramework/Analyses/UnsafeBufferUsage/UnsafeBufferUsageAnalysis.h"
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

UnsafeBufferUsageEntitySummary
ssaf::buildUnsafeBufferUsageEntitySummary(EntityPointerLevelSet UnsafeBuffers);

namespace {

class UnsafeBufferUsageWPATest : public TestFixture {
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

  void insertSummary(LUSummary &LU, EntityId Id, EntityPointerLevelSet EPLs) {
    getData(LU)[UnsafeBufferUsageEntitySummary::summaryName()][Id] =
        std::make_unique<UnsafeBufferUsageEntitySummary>(
            buildUnsafeBufferUsageEntitySummary(std::move(EPLs)));
  }

  const UnsafeBufferUsageAnalysisResult &
  runAnalysisForResult(std::unique_ptr<LUSummary> LU) {
    AnalysisDriver Driver(std::move(LU));
    auto WPAOrErr = Driver.run<UnsafeBufferUsageAnalysisResult>();
    EXPECT_THAT_EXPECTED(WPAOrErr, llvm::Succeeded());
    WPA = std::move(*WPAOrErr);
    auto ROrErr = WPA.get<UnsafeBufferUsageAnalysisResult>();
    EXPECT_THAT_EXPECTED(ROrErr, llvm::Succeeded());
    return *ROrErr;
  }

private:
  WPASuite WPA = makeWPASuite();
};

TEST_F(UnsafeBufferUsageWPATest, SingleEntitySummary) {
  auto LU = makeLUSummary();
  EntityId Foo = addEntity(*LU, "c:@F@foo");
  EntityId P = addEntity(*LU, "c:@F@foo@p");

  EntityPointerLevelSet EPLs{buildEntityPointerLevel(P, 1),
                             buildEntityPointerLevel(P, 2),
                             buildEntityPointerLevel(P, 3)};
  insertSummary(*LU, Foo, EPLs);

  const auto &R = runAnalysisForResult(std::move(LU));

  ASSERT_EQ(R.UnsafeBuffers.count(Foo), 1u);
  EXPECT_THAT(R.UnsafeBuffers.at(Foo),
              UnorderedElementsAre(buildEntityPointerLevel(P, 1),
                                   buildEntityPointerLevel(P, 2),
                                   buildEntityPointerLevel(P, 3)));
}

TEST_F(UnsafeBufferUsageWPATest, MultiEntitySummaries) {
  auto LU = makeLUSummary();
  EntityId Foo = addEntity(*LU, "c:@F@foo");
  EntityId Bar = addEntity(*LU, "c:@F@bar");
  EntityId P = addEntity(*LU, "c:@F@foo@p");
  EntityId Q = addEntity(*LU, "c:@F@bar@q");

  insertSummary(*LU, Foo,
                {buildEntityPointerLevel(P, 1), buildEntityPointerLevel(P, 2),
                 buildEntityPointerLevel(P, 3)});
  insertSummary(*LU, Bar,
                {buildEntityPointerLevel(Q, 1), buildEntityPointerLevel(Q, 2)});

  const auto &R = runAnalysisForResult(std::move(LU));

  EXPECT_EQ(R.UnsafeBuffers.size(), 2u);
  EXPECT_THAT(R.UnsafeBuffers.at(Foo),
              UnorderedElementsAre(buildEntityPointerLevel(P, 1),
                                   buildEntityPointerLevel(P, 2),
                                   buildEntityPointerLevel(P, 3)));
  EXPECT_THAT(R.UnsafeBuffers.at(Bar),
              UnorderedElementsAre(buildEntityPointerLevel(Q, 1),
                                   buildEntityPointerLevel(Q, 2)));
}

TEST_F(UnsafeBufferUsageWPATest, EmptySummary) {
  auto LU = makeLUSummary();
  EntityId Foo = addEntity(*LU, "c:@F@foo");

  insertSummary(*LU, Foo, {});

  const auto &R = runAnalysisForResult(std::move(LU));

  ASSERT_EQ(R.UnsafeBuffers.count(Foo), 1u);
  EXPECT_THAT(R.UnsafeBuffers.at(Foo), IsEmpty());
}
} // namespace
