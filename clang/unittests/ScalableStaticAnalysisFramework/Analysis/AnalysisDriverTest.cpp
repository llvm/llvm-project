//===- AnalysisDriverTest.cpp ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/ScalableStaticAnalysisFramework/Core/Analysis/AnalysisDriver.h"
#include "../TestFixture.h"
#include "clang/ScalableStaticAnalysisFramework/Core/Analysis/AnalysisName.h"
#include "clang/ScalableStaticAnalysisFramework/Core/Analysis/AnalysisRegistry.h"
#include "clang/ScalableStaticAnalysisFramework/Core/Analysis/AnalysisResult.h"
#include "clang/ScalableStaticAnalysisFramework/Core/Analysis/SummaryAnalysis.h"
#include "clang/ScalableStaticAnalysisFramework/Core/Analysis/WPASuite.h"
#include "clang/ScalableStaticAnalysisFramework/Core/EntityLinker/LUSummary.h"
#include "clang/ScalableStaticAnalysisFramework/Core/Model/BuildNamespace.h"
#include "clang/ScalableStaticAnalysisFramework/Core/Model/EntityId.h"
#include "clang/ScalableStaticAnalysisFramework/Core/Model/EntityLinkage.h"
#include "clang/ScalableStaticAnalysisFramework/Core/Model/EntityName.h"
#include "clang/ScalableStaticAnalysisFramework/Core/Model/SummaryName.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"
#include <memory>
#include <utility>
#include <vector>

using namespace clang;
using namespace ssaf;

namespace {

// ---------------------------------------------------------------------------
// Instance counter
// ---------------------------------------------------------------------------

static int NextSummaryInstanceId = 0;

// ---------------------------------------------------------------------------
// Entity summaries
// ---------------------------------------------------------------------------

class Analysis1EntitySummary final : public EntitySummary {
public:
  int InstanceId = NextSummaryInstanceId++;
  static SummaryName summaryName() { return SummaryName("Analysis1"); }
  SummaryName getSummaryName() const override {
    return SummaryName("Analysis1");
  }
};

class Analysis2EntitySummary final : public EntitySummary {
public:
  int InstanceId = NextSummaryInstanceId++;
  static SummaryName summaryName() { return SummaryName("Analysis2"); }
  SummaryName getSummaryName() const override {
    return SummaryName("Analysis2");
  }
};

class Analysis3EntitySummary final : public EntitySummary {
public:
  int InstanceId = NextSummaryInstanceId++;
  static SummaryName summaryName() { return SummaryName("Analysis3"); }
  SummaryName getSummaryName() const override {
    return SummaryName("Analysis3");
  }
};

class Analysis4EntitySummary final : public EntitySummary {
public:
  int InstanceId = NextSummaryInstanceId++;
  static SummaryName summaryName() { return SummaryName("Analysis4"); }
  SummaryName getSummaryName() const override {
    return SummaryName("Analysis4");
  }
};

// ---------------------------------------------------------------------------
// Results
// ---------------------------------------------------------------------------

class Analysis1Result final : public AnalysisResult {
public:
  static AnalysisName analysisName() { return AnalysisName("Analysis1"); }
  std::vector<std::pair<EntityId, int>> Entries;
  bool WasFinalized = false;
};

class Analysis2Result final : public AnalysisResult {
public:
  static AnalysisName analysisName() { return AnalysisName("Analysis2"); }
  std::vector<std::pair<EntityId, int>> Entries;
  bool WasFinalized = false;
};

// No analysis or registration for Analysis3. Data for Analysis3 is inserted
// into the LUSummary to verify the driver silently skips it.
class Analysis3Result final : public AnalysisResult {
public:
  static AnalysisName analysisName() { return AnalysisName("Analysis3"); }
};

// Analysis4 has a registered analysis but no data is inserted into the
// LUSummary, so it is skipped and get() returns nullptr.
class Analysis4Result final : public AnalysisResult {
public:
  static AnalysisName analysisName() { return AnalysisName("Analysis4"); }
  std::vector<std::pair<EntityId, int>> Entries;
  bool WasFinalized = false;
};

// ---------------------------------------------------------------------------
// Analysis destruction flags (reset in SetUp)
// ---------------------------------------------------------------------------

static bool Analysis1WasDestroyed = false;
static bool Analysis2WasDestroyed = false;
static bool Analysis4WasDestroyed = false;

// ---------------------------------------------------------------------------
// Analyses
// ---------------------------------------------------------------------------

class Analysis1 final
    : public SummaryAnalysis<Analysis1Result, Analysis1EntitySummary> {
public:
  ~Analysis1() { Analysis1WasDestroyed = true; }

  llvm::Error add(EntityId Id, const Analysis1EntitySummary &S) override {
    result().Entries.push_back({Id, S.InstanceId});
    return llvm::Error::success();
  }

  llvm::Error finalize() override {
    result().WasFinalized = true;
    return llvm::Error::success();
  }
};

static AnalysisRegistry::Add<Analysis1> RegAnalysis1("Analysis for Analysis1");

class Analysis2 final
    : public SummaryAnalysis<Analysis2Result, Analysis2EntitySummary> {
public:
  ~Analysis2() { Analysis2WasDestroyed = true; }

  llvm::Error add(EntityId Id, const Analysis2EntitySummary &S) override {
    result().Entries.push_back({Id, S.InstanceId});
    return llvm::Error::success();
  }

  llvm::Error finalize() override {
    result().WasFinalized = true;
    return llvm::Error::success();
  }
};

static AnalysisRegistry::Add<Analysis2> RegAnalysis2("Analysis for Analysis2");

class Analysis4 final
    : public SummaryAnalysis<Analysis4Result, Analysis4EntitySummary> {
public:
  ~Analysis4() { Analysis4WasDestroyed = true; }

  llvm::Error add(EntityId Id, const Analysis4EntitySummary &S) override {
    result().Entries.push_back({Id, S.InstanceId});
    return llvm::Error::success();
  }

  llvm::Error finalize() override {
    result().WasFinalized = true;
    return llvm::Error::success();
  }
};

static AnalysisRegistry::Add<Analysis4> RegAnalysis4("Analysis for Analysis4");

// ---------------------------------------------------------------------------
// Fixture
// ---------------------------------------------------------------------------

class AnalysisDriverTest : public TestFixture {
protected:
  static constexpr EntityLinkage ExternalLinkage =
      EntityLinkage(EntityLinkageType::External);

  void SetUp() override {
    NextSummaryInstanceId = 0;
    Analysis1WasDestroyed = false;
    Analysis2WasDestroyed = false;
    Analysis4WasDestroyed = false;
  }

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

  static bool hasEntry(const std::vector<std::pair<EntityId, int>> &Entries,
                       EntityId Id, int InstanceId) {
    return llvm::is_contained(Entries, std::make_pair(Id, InstanceId));
  }

  template <typename SummaryT>
  int insertSummary(LUSummary &LU, llvm::StringRef SN, EntityId Id) {
    auto S = std::make_unique<SummaryT>();
    int InstanceId = S->InstanceId;
    getData(LU)[SummaryName(SN.str())][Id] = std::move(S);
    return InstanceId;
  }
};

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

TEST(AnalysisRegistryTest, AnalysisIsRegistered) {
  EXPECT_FALSE(AnalysisRegistry::contains("AnalysisNonExisting"));
  EXPECT_TRUE(AnalysisRegistry::contains("Analysis1"));
  EXPECT_TRUE(AnalysisRegistry::contains("Analysis2"));
  EXPECT_TRUE(AnalysisRegistry::contains("Analysis4"));
}

TEST(AnalysisRegistryTest, AnalysisCanBeInstantiated) {
  EXPECT_THAT_EXPECTED(AnalysisRegistry::instantiate("AnalysisNonExisting"),
                       llvm::Failed());
  EXPECT_THAT_EXPECTED(AnalysisRegistry::instantiate("Analysis1"),
                       llvm::Succeeded());
  EXPECT_THAT_EXPECTED(AnalysisRegistry::instantiate("Analysis2"),
                       llvm::Succeeded());
  EXPECT_THAT_EXPECTED(AnalysisRegistry::instantiate("Analysis4"),
                       llvm::Succeeded());
}

// run() — processes all registered analyses present in the LUSummary.
// Silently skips data whose analysis is unregistered (Analysis3).
TEST_F(AnalysisDriverTest, RunAll) {
  auto LU = makeLUSummary();
  const auto E1 = addEntity(*LU, "Entity1");
  const auto E2 = addEntity(*LU, "Entity2");
  const auto E3 = addEntity(*LU, "Entity3");
  const auto E4 = addEntity(*LU, "Entity4");

  int s1a = insertSummary<Analysis1EntitySummary>(*LU, "Analysis1", E1);
  int s1b = insertSummary<Analysis1EntitySummary>(*LU, "Analysis1", E2);
  int s2a = insertSummary<Analysis2EntitySummary>(*LU, "Analysis2", E2);
  int s2b = insertSummary<Analysis2EntitySummary>(*LU, "Analysis2", E3);
  int s4a = insertSummary<Analysis4EntitySummary>(*LU, "Analysis4", E4);

  // No registered analysis — Analysis3 data silently skipped.
  (void)insertSummary<Analysis3EntitySummary>(*LU, "Analysis3", E1);

  AnalysisDriver Driver(std::move(LU));
  auto WPAOrErr = std::move(Driver).run();
  ASSERT_THAT_EXPECTED(WPAOrErr, llvm::Succeeded());

  {
    auto R1OrErr = WPAOrErr->get<Analysis1Result>();
    ASSERT_THAT_EXPECTED(R1OrErr, llvm::Succeeded());
    EXPECT_EQ(R1OrErr->Entries.size(), 2u);
    EXPECT_TRUE(hasEntry(R1OrErr->Entries, E1, s1a));
    EXPECT_TRUE(hasEntry(R1OrErr->Entries, E2, s1b));
    EXPECT_TRUE(R1OrErr->WasFinalized);
    EXPECT_TRUE(Analysis1WasDestroyed);
  }

  {
    auto R2OrErr = WPAOrErr->get<Analysis2Result>();
    ASSERT_THAT_EXPECTED(R2OrErr, llvm::Succeeded());
    EXPECT_EQ(R2OrErr->Entries.size(), 2u);
    EXPECT_TRUE(hasEntry(R2OrErr->Entries, E2, s2a));
    EXPECT_TRUE(hasEntry(R2OrErr->Entries, E3, s2b));
    EXPECT_TRUE(R2OrErr->WasFinalized);
    EXPECT_TRUE(Analysis2WasDestroyed);
  }

  {
    auto R4OrErr = WPAOrErr->get<Analysis4Result>();
    ASSERT_THAT_EXPECTED(R4OrErr, llvm::Succeeded());
    EXPECT_EQ(R4OrErr->Entries.size(), 1u);
    EXPECT_TRUE(hasEntry(R4OrErr->Entries, E4, s4a));
    EXPECT_TRUE(R4OrErr->WasFinalized);
    EXPECT_TRUE(Analysis4WasDestroyed);
  }

  // Unregistered analysis — not present in WPA.
  EXPECT_THAT_EXPECTED(WPAOrErr->get<Analysis3Result>(), llvm::Failed());
}

// run(names) — processes only the analyses for the given names.
TEST_F(AnalysisDriverTest, RunByName) {
  auto LU = makeLUSummary();
  const auto E1 = addEntity(*LU, "Entity1");
  const auto E2 = addEntity(*LU, "Entity2");

  int s1a = insertSummary<Analysis1EntitySummary>(*LU, "Analysis1", E1);
  insertSummary<Analysis2EntitySummary>(*LU, "Analysis2", E2);

  AnalysisDriver Driver(std::move(LU));
  auto WPAOrErr = Driver.run({AnalysisName("Analysis1")});
  ASSERT_THAT_EXPECTED(WPAOrErr, llvm::Succeeded());

  // Analysis1 was requested and has data — present.
  auto R1OrErr = WPAOrErr->get<Analysis1Result>();
  ASSERT_THAT_EXPECTED(R1OrErr, llvm::Succeeded());
  EXPECT_EQ(R1OrErr->Entries.size(), 1u);
  EXPECT_TRUE(hasEntry(R1OrErr->Entries, E1, s1a));
  EXPECT_TRUE(R1OrErr->WasFinalized);

  // Analysis2 was not requested — not present even though data exists.
  EXPECT_THAT_EXPECTED(WPAOrErr->get<Analysis2Result>(), llvm::Failed());
}

// run(names) — error when a requested name has no data in LUSummary.
TEST_F(AnalysisDriverTest, RunByNameErrorMissingData) {
  auto LU = makeLUSummary();
  AnalysisDriver Driver(std::move(LU));

  EXPECT_THAT_EXPECTED(Driver.run({AnalysisName("Analysis1")}), llvm::Failed());
}

// run(names) — error when a requested name has no registered analysis.
TEST_F(AnalysisDriverTest, RunByNameErrorMissingAnalysis) {
  auto LU = makeLUSummary();
  const auto E1 = addEntity(*LU, "Entity1");
  insertSummary<Analysis3EntitySummary>(*LU, "Analysis3", E1);

  AnalysisDriver Driver(std::move(LU));

  // Analysis3 has data but no registered analysis.
  EXPECT_THAT_EXPECTED(Driver.run({AnalysisName("Analysis3")}), llvm::Failed());
}

// run<ResultTs...>() — type-safe subset.
TEST_F(AnalysisDriverTest, RunByType) {
  auto LU = makeLUSummary();
  const auto E1 = addEntity(*LU, "Entity1");
  const auto E2 = addEntity(*LU, "Entity2");

  int s1a = insertSummary<Analysis1EntitySummary>(*LU, "Analysis1", E1);
  insertSummary<Analysis2EntitySummary>(*LU, "Analysis2", E2);

  AnalysisDriver Driver(std::move(LU));
  auto WPAOrErr = Driver.run<Analysis1Result>();
  ASSERT_THAT_EXPECTED(WPAOrErr, llvm::Succeeded());

  // Analysis1 was requested — present.
  auto R1OrErr = WPAOrErr->get<Analysis1Result>();
  ASSERT_THAT_EXPECTED(R1OrErr, llvm::Succeeded());
  EXPECT_EQ(R1OrErr->Entries.size(), 1u);
  EXPECT_TRUE(hasEntry(R1OrErr->Entries, E1, s1a));
  EXPECT_TRUE(R1OrErr->WasFinalized);

  // Analysis2 was not requested — not present even though data exists.
  EXPECT_THAT_EXPECTED(WPAOrErr->get<Analysis2Result>(), llvm::Failed());
}

// run<ResultTs...>() — error when a requested type has no data in LUSummary.
TEST_F(AnalysisDriverTest, RunByTypeErrorMissingData) {
  auto LU = makeLUSummary();
  AnalysisDriver Driver(std::move(LU));

  EXPECT_THAT_EXPECTED(Driver.run<Analysis1Result>(), llvm::Failed());
}

// contains() — present entries return true; absent entries return false.
TEST_F(AnalysisDriverTest, Contains) {
  auto LU = makeLUSummary();
  const auto E1 = addEntity(*LU, "Entity1");
  insertSummary<Analysis1EntitySummary>(*LU, "Analysis1", E1);
  insertSummary<Analysis4EntitySummary>(*LU, "Analysis4", E1);

  AnalysisDriver Driver(std::move(LU));
  auto WPAOrErr = std::move(Driver).run();
  ASSERT_THAT_EXPECTED(WPAOrErr, llvm::Succeeded());

  EXPECT_TRUE(WPAOrErr->contains<Analysis1Result>());
  EXPECT_FALSE(WPAOrErr->contains<Analysis2Result>());
}

} // namespace
