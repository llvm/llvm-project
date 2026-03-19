//===- AnalysisDriverTest.cpp ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/ScalableStaticAnalysisFramework/Core/WholeProgramAnalysis/AnalysisDriver.h"
#include "../TestFixture.h"
#include "clang/ScalableStaticAnalysisFramework/Core/EntityLinker/LUSummary.h"
#include "clang/ScalableStaticAnalysisFramework/Core/Model/BuildNamespace.h"
#include "clang/ScalableStaticAnalysisFramework/Core/Model/EntityId.h"
#include "clang/ScalableStaticAnalysisFramework/Core/Model/EntityLinkage.h"
#include "clang/ScalableStaticAnalysisFramework/Core/Model/EntityName.h"
#include "clang/ScalableStaticAnalysisFramework/Core/Model/SummaryName.h"
#include "clang/ScalableStaticAnalysisFramework/Core/WholeProgramAnalysis/AnalysisName.h"
#include "clang/ScalableStaticAnalysisFramework/Core/WholeProgramAnalysis/AnalysisRegistry.h"
#include "clang/ScalableStaticAnalysisFramework/Core/WholeProgramAnalysis/AnalysisResult.h"
#include "clang/ScalableStaticAnalysisFramework/Core/WholeProgramAnalysis/DerivedAnalysis.h"
#include "clang/ScalableStaticAnalysisFramework/Core/WholeProgramAnalysis/SummaryAnalysis.h"
#include "clang/ScalableStaticAnalysisFramework/Core/WholeProgramAnalysis/WPASuite.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"
#include <memory>
#include <string>
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
// Analysis names
// ---------------------------------------------------------------------------

const AnalysisName Analysis1Name("Analysis1");
const AnalysisName Analysis2Name("Analysis2");
const AnalysisName Analysis3Name("Analysis3");
const AnalysisName Analysis4Name("Analysis4");
const AnalysisName Analysis5Name("Analysis5");
const AnalysisName CycleAName("CycleA");
const AnalysisName CycleBName("CycleB");

// ---------------------------------------------------------------------------
// Results
// ---------------------------------------------------------------------------

class Analysis1Result final : public AnalysisResult {
public:
  static AnalysisName analysisName() { return Analysis1Name; }
  std::vector<std::pair<EntityId, int>> Entries;
  bool WasInitialized = false;
  bool WasFinalized = false;
};

class Analysis2Result final : public AnalysisResult {
public:
  static AnalysisName analysisName() { return Analysis2Name; }
  std::vector<std::pair<EntityId, int>> Entries;
  bool WasInitialized = false;
  bool WasFinalized = false;
};

// No analysis or registration for Analysis3. Data for Analysis3 is inserted
// into the LUSummary to verify the driver silently skips it.
class Analysis3Result final : public AnalysisResult {
public:
  static AnalysisName analysisName() { return Analysis3Name; }
};

// Analysis4 has a registered analysis but no data is inserted into the
// LUSummary, so it is skipped and get() returns nullptr.
class Analysis4Result final : public AnalysisResult {
public:
  static AnalysisName analysisName() { return Analysis4Name; }
  std::vector<std::pair<EntityId, int>> Entries;
  bool WasInitialized = false;
  bool WasFinalized = false;
};

// Analysis5 is a derived analysis that depends on Analysis1, Analysis2, and
// Analysis4. It verifies that the driver passes dependency results to
// initialize() and that the initialize/step/finalize lifecycle is respected.
class Analysis5Result final : public AnalysisResult {
public:
  static AnalysisName analysisName() { return Analysis5Name; }
  std::vector<std::string> CallSequence;
  std::vector<std::pair<EntityId, int>> Analysis1Entries;
  std::vector<std::pair<EntityId, int>> Analysis2Entries;
  std::vector<std::pair<EntityId, int>> Analysis4Entries;
};

// CycleA and CycleB form a dependency cycle (CycleA → CycleB → CycleA).
// Registered solely to exercise cycle detection in AnalysisDriver::toposort().
// initialize() and step() are unreachable stubs - the cycle is caught before
// any analysis executes.
class CycleAResult final : public AnalysisResult {
public:
  static AnalysisName analysisName() { return CycleAName; }
};

class CycleBResult final : public AnalysisResult {
public:
  static AnalysisName analysisName() { return CycleBName; }
};

// ---------------------------------------------------------------------------
// Analyses
// ---------------------------------------------------------------------------

class Analysis1 final
    : public SummaryAnalysis<Analysis1Result, Analysis1EntitySummary> {
public:
  inline static bool WasDestroyed = false;
  ~Analysis1() { WasDestroyed = true; }

  llvm::Error initialize() override {
    result().WasInitialized = true;
    return llvm::Error::success();
  }

  llvm::Error add(EntityId Id, const Analysis1EntitySummary &S) override {
    result().Entries.push_back({Id, S.InstanceId});
    return llvm::Error::success();
  }

  llvm::Error finalize() override {
    result().WasFinalized = true;
    return llvm::Error::success();
  }
};

// These static registrations are safe without SSAFBuiltinTestForceLinker.h
// because this translation unit is compiled directly into the test binary -
// the linker cannot dead-strip it, so all static initializers are guaranteed
// to run.
static AnalysisRegistry::Add<Analysis1> RegAnalysis1("Analysis for Analysis1");

class Analysis2 final
    : public SummaryAnalysis<Analysis2Result, Analysis2EntitySummary> {
public:
  inline static bool WasDestroyed = false;
  ~Analysis2() { WasDestroyed = true; }

  llvm::Error initialize() override {
    result().WasInitialized = true;
    return llvm::Error::success();
  }

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

// No Analysis3 or registration for Analysis3.

class Analysis4 final
    : public SummaryAnalysis<Analysis4Result, Analysis4EntitySummary> {
public:
  inline static bool WasDestroyed = false;
  ~Analysis4() { WasDestroyed = true; }

  llvm::Error initialize() override {
    result().WasInitialized = true;
    return llvm::Error::success();
  }

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

class Analysis5 final
    : public DerivedAnalysis<Analysis5Result, Analysis1Result, Analysis2Result,
                             Analysis4Result> {
  int StepCount = 0;

public:
  inline static bool WasDestroyed = false;
  ~Analysis5() { WasDestroyed = true; }

  llvm::Error initialize(const Analysis1Result &R1, const Analysis2Result &R2,
                         const Analysis4Result &R4) override {
    result().CallSequence.push_back("initialize");
    result().Analysis1Entries = R1.Entries;
    result().Analysis2Entries = R2.Entries;
    result().Analysis4Entries = R4.Entries;
    return llvm::Error::success();
  }

  llvm::Expected<bool> step() override {
    result().CallSequence.push_back("step");
    return ++StepCount < 2;
  }

  llvm::Error finalize() override {
    result().CallSequence.push_back("finalize");
    return llvm::Error::success();
  }
};

static AnalysisRegistry::Add<Analysis5> RegAnalysis5("Analysis for Analysis5");

class CycleA final : public DerivedAnalysis<CycleAResult, CycleBResult> {
public:
  llvm::Error initialize(const CycleBResult &) override {
    return llvm::Error::success();
  }
  llvm::Expected<bool> step() override { return false; }
};

static AnalysisRegistry::Add<CycleA> RegCycleA("Cyclic analysis A (test only)");

class CycleB final : public DerivedAnalysis<CycleBResult, CycleAResult> {
public:
  llvm::Error initialize(const CycleAResult &) override {
    return llvm::Error::success();
  }
  llvm::Expected<bool> step() override { return false; }
};

static AnalysisRegistry::Add<CycleB> RegCycleB("Cyclic analysis B (test only)");

// ---------------------------------------------------------------------------
// Fixture
// ---------------------------------------------------------------------------

class AnalysisDriverTest : public TestFixture {
protected:
  static constexpr EntityLinkage ExternalLinkage =
      EntityLinkage(EntityLinkageType::External);

  void SetUp() override {
    NextSummaryInstanceId = 0;
    Analysis1::WasDestroyed = false;
    Analysis2::WasDestroyed = false;
    // No Analysis3 - not registered, so no WasDestroyed flag.
    Analysis4::WasDestroyed = false;
    Analysis5::WasDestroyed = false;
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
  EXPECT_TRUE(AnalysisRegistry::contains(Analysis1Name));
  EXPECT_TRUE(AnalysisRegistry::contains(Analysis2Name));
  EXPECT_FALSE(AnalysisRegistry::contains(Analysis3Name));
  EXPECT_TRUE(AnalysisRegistry::contains(Analysis4Name));
  EXPECT_TRUE(AnalysisRegistry::contains(Analysis5Name));
  EXPECT_TRUE(AnalysisRegistry::contains(CycleAName));
  EXPECT_TRUE(AnalysisRegistry::contains(CycleBName));
}

TEST(AnalysisRegistryTest, AnalysisCanBeInstantiated) {
  constexpr auto instantiate = AnalysisRegistry::instantiate;
  EXPECT_THAT_EXPECTED(
      instantiate(AnalysisName("AnalysisNonExisting")),
      llvm::FailedWithMessage(
          "no analysis registered for 'AnalysisName(AnalysisNonExisting)'"));
  EXPECT_THAT_EXPECTED(instantiate(Analysis1Name), llvm::Succeeded());
  EXPECT_THAT_EXPECTED(instantiate(Analysis2Name), llvm::Succeeded());
  // No Analysis3 - not registered, so instantiate() would fail.
  EXPECT_THAT_EXPECTED(instantiate(Analysis4Name), llvm::Succeeded());
  EXPECT_THAT_EXPECTED(instantiate(Analysis5Name), llvm::Succeeded());
  EXPECT_THAT_EXPECTED(instantiate(CycleAName), llvm::Succeeded());
  EXPECT_THAT_EXPECTED(instantiate(CycleBName), llvm::Succeeded());
}

// run<T...>() — processes the non-cyclic analyses in topological order.
// CycleA and CycleB are excluded because they form a cycle; run() && would
// error on them, so the type-safe subset overload is used here instead.
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
  auto WPAOrErr = Driver.run<Analysis1Result, Analysis2Result, Analysis4Result,
                             Analysis5Result>();
  ASSERT_THAT_EXPECTED(WPAOrErr, llvm::Succeeded());

  {
    auto R1OrErr = WPAOrErr->get<Analysis1Result>();
    ASSERT_THAT_EXPECTED(R1OrErr, llvm::Succeeded());
    EXPECT_EQ(R1OrErr->Entries.size(), 2u);
    EXPECT_TRUE(hasEntry(R1OrErr->Entries, E1, s1a));
    EXPECT_TRUE(hasEntry(R1OrErr->Entries, E2, s1b));
    EXPECT_TRUE(R1OrErr->WasInitialized);
    EXPECT_TRUE(R1OrErr->WasFinalized);
    EXPECT_TRUE(Analysis1::WasDestroyed);
  }

  {
    auto R2OrErr = WPAOrErr->get<Analysis2Result>();
    ASSERT_THAT_EXPECTED(R2OrErr, llvm::Succeeded());
    EXPECT_EQ(R2OrErr->Entries.size(), 2u);
    EXPECT_TRUE(hasEntry(R2OrErr->Entries, E2, s2a));
    EXPECT_TRUE(hasEntry(R2OrErr->Entries, E3, s2b));
    EXPECT_TRUE(R2OrErr->WasInitialized);
    EXPECT_TRUE(R2OrErr->WasFinalized);
    EXPECT_TRUE(Analysis2::WasDestroyed);
  }

  {
    auto R4OrErr = WPAOrErr->get<Analysis4Result>();
    ASSERT_THAT_EXPECTED(R4OrErr, llvm::Succeeded());
    EXPECT_EQ(R4OrErr->Entries.size(), 1u);
    EXPECT_TRUE(hasEntry(R4OrErr->Entries, E4, s4a));
    EXPECT_TRUE(R4OrErr->WasInitialized);
    EXPECT_TRUE(R4OrErr->WasFinalized);
    EXPECT_TRUE(Analysis4::WasDestroyed);
  }

  {
    auto R5OrErr = WPAOrErr->get<Analysis5Result>();
    ASSERT_THAT_EXPECTED(R5OrErr, llvm::Succeeded());
    EXPECT_EQ(
        R5OrErr->CallSequence,
        (std::vector<std::string>{"initialize", "step", "step", "finalize"}));
    EXPECT_EQ(R5OrErr->Analysis1Entries.size(), 2u);
    EXPECT_TRUE(hasEntry(R5OrErr->Analysis1Entries, E1, s1a));
    EXPECT_TRUE(hasEntry(R5OrErr->Analysis1Entries, E2, s1b));
    EXPECT_EQ(R5OrErr->Analysis2Entries.size(), 2u);
    EXPECT_TRUE(hasEntry(R5OrErr->Analysis2Entries, E2, s2a));
    EXPECT_TRUE(hasEntry(R5OrErr->Analysis2Entries, E3, s2b));
    EXPECT_EQ(R5OrErr->Analysis4Entries.size(), 1u);
    EXPECT_TRUE(hasEntry(R5OrErr->Analysis4Entries, E4, s4a));
    EXPECT_TRUE(Analysis5::WasDestroyed);
  }

  // Unregistered analysis — not present in WPA.
  EXPECT_THAT_EXPECTED(
      WPAOrErr->get<Analysis3Result>(),
      llvm::FailedWithMessage(
          "no result for 'AnalysisName(Analysis3)' in WPASuite"));
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
  EXPECT_TRUE(R1OrErr->WasInitialized);
  EXPECT_TRUE(R1OrErr->WasFinalized);

  // Analysis2 was not requested — not present even though data exists.
  EXPECT_THAT_EXPECTED(
      WPAOrErr->get<Analysis2Result>(),
      llvm::FailedWithMessage(
          "no result for 'AnalysisName(Analysis2)' in WPASuite"));
}

// run(names) — error when a requested name has no data in LUSummary.
TEST_F(AnalysisDriverTest, RunByNameErrorMissingData) {
  auto LU = makeLUSummary();
  AnalysisDriver Driver(std::move(LU));

  EXPECT_THAT_EXPECTED(
      Driver.run({AnalysisName("Analysis1")}),
      llvm::FailedWithMessage(
          "no data for analysis 'AnalysisName(Analysis1)' in LUSummary"));
}

// run(names) — error when a requested name has no registered analysis.
TEST_F(AnalysisDriverTest, RunByNameErrorMissingAnalysis) {
  auto LU = makeLUSummary();
  const auto E1 = addEntity(*LU, "Entity1");
  insertSummary<Analysis3EntitySummary>(*LU, "Analysis3", E1);

  AnalysisDriver Driver(std::move(LU));

  // Analysis3 has data but no registered analysis.
  EXPECT_THAT_EXPECTED(
      Driver.run({AnalysisName("Analysis3")}),
      llvm::FailedWithMessage(
          "no analysis registered for 'AnalysisName(Analysis3)'"));
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
  EXPECT_TRUE(R1OrErr->WasInitialized);
  EXPECT_TRUE(R1OrErr->WasFinalized);

  // Analysis2 was not requested — not present even though data exists.
  EXPECT_THAT_EXPECTED(
      WPAOrErr->get<Analysis2Result>(),
      llvm::FailedWithMessage(
          "no result for 'AnalysisName(Analysis2)' in WPASuite"));
}

// run<ResultTs...>() — error when a requested type has no data in LUSummary.
TEST_F(AnalysisDriverTest, RunByTypeErrorMissingData) {
  auto LU = makeLUSummary();
  AnalysisDriver Driver(std::move(LU));

  EXPECT_THAT_EXPECTED(
      Driver.run<Analysis1Result>(),
      llvm::FailedWithMessage(
          "no data for analysis 'AnalysisName(Analysis1)' in LUSummary"));
}

// contains() — present entries return true; absent entries return false.
TEST_F(AnalysisDriverTest, Contains) {
  auto LU = makeLUSummary();
  const auto E1 = addEntity(*LU, "Entity1");
  insertSummary<Analysis1EntitySummary>(*LU, "Analysis1", E1);
  insertSummary<Analysis2EntitySummary>(*LU, "Analysis2", E1);
  insertSummary<Analysis4EntitySummary>(*LU, "Analysis4", E1);

  AnalysisDriver Driver(std::move(LU));
  auto WPAOrErr = Driver.run<Analysis1Result, Analysis2Result, Analysis4Result,
                             Analysis5Result>();
  ASSERT_THAT_EXPECTED(WPAOrErr, llvm::Succeeded());
  EXPECT_TRUE(WPAOrErr->contains<Analysis1Result>());
  // Analysis3 has no registered analysis — never present in WPA.
  EXPECT_FALSE(WPAOrErr->contains<Analysis3Result>());
}

// run() && — errors when the registry contains a dependency cycle.
TEST_F(AnalysisDriverTest, CycleDetected) {
  auto LU = makeLUSummary();
  AnalysisDriver Driver(std::move(LU));
  EXPECT_THAT_EXPECTED(
      std::move(Driver).run(),
      llvm::FailedWithMessage("cycle detected: AnalysisName(CycleA) -> "
                              "AnalysisName(CycleB) -> AnalysisName(CycleA)"));
}

} // namespace
