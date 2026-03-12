//===- SummaryDataTest.cpp ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../TestFixture.h"
#include "clang/Analysis/Scalable/EntityLinker/LUSummary.h"
#include "clang/Analysis/Scalable/Model/BuildNamespace.h"
#include "clang/Analysis/Scalable/Model/EntityId.h"
#include "clang/Analysis/Scalable/Model/EntityIdTable.h"
#include "clang/Analysis/Scalable/Model/EntityLinkage.h"
#include "clang/Analysis/Scalable/Model/EntityName.h"
#include "clang/Analysis/Scalable/Model/SummaryName.h"
#include "clang/Analysis/Scalable/SummaryData/LUSummaryConsumer.h"
#include "clang/Analysis/Scalable/SummaryData/SummaryDataBuilderRegistry.h"
#include "clang/Analysis/Scalable/TUSummary/EntitySummary.h"
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
  SummaryName getSummaryName() const override {
    return SummaryName("Analysis1");
  }
};

class Analysis2EntitySummary final : public EntitySummary {
public:
  int InstanceId = NextSummaryInstanceId++;
  SummaryName getSummaryName() const override {
    return SummaryName("Analysis2");
  }
};

class Analysis3EntitySummary final : public EntitySummary {
public:
  int InstanceId = NextSummaryInstanceId++;
  SummaryName getSummaryName() const override {
    return SummaryName("Analysis3");
  }
};

class Analysis4EntitySummary final : public EntitySummary {
public:
  int InstanceId = NextSummaryInstanceId++;
  SummaryName getSummaryName() const override {
    return SummaryName("Analysis4");
  }
};

// ---------------------------------------------------------------------------
// Data
// ---------------------------------------------------------------------------

class Analysis1Data final : public SummaryData {
public:
  static SummaryName summaryName() { return SummaryName("Analysis1"); }
  std::vector<std::pair<EntityId, int>> Entries;
  bool WasFinalized = false;
};

class Analysis2Data final : public SummaryData {
public:
  static SummaryName summaryName() { return SummaryName("Analysis2"); }
  std::vector<std::pair<EntityId, int>> Entries;
  bool WasFinalized = false;
};

// No builder or registration for Analysis3. Data for Analysis3 is inserted
// into the LUSummary to verify the consumer silently skips it.
class Analysis3Data final : public SummaryData {
public:
  static SummaryName summaryName() { return SummaryName("Analysis3"); }
};

// Analysis4 has a registered builder but no data is inserted into the
// LUSummary, so the builder is never invoked and getData returns nullptr.
class Analysis4Data final : public SummaryData {
public:
  static SummaryName summaryName() { return SummaryName("Analysis4"); }
  std::vector<std::pair<EntityId, int>> Entries;
  bool WasFinalized = false;
};

// ---------------------------------------------------------------------------
// Builder destruction flags (reset in SetUp)
// ---------------------------------------------------------------------------

static bool Analysis1BuilderWasDestroyed = false;
static bool Analysis2BuilderWasDestroyed = false;
static bool Analysis4BuilderWasDestroyed = false;

// ---------------------------------------------------------------------------
// Builders
// ---------------------------------------------------------------------------

class Analysis1Builder final
    : public SummaryDataBuilder<Analysis1Data, Analysis1EntitySummary> {
public:
  ~Analysis1Builder() { Analysis1BuilderWasDestroyed = true; }

  void addSummary(EntityId Id,
                  std::unique_ptr<Analysis1EntitySummary> S) override {
    getData().Entries.push_back({Id, S->InstanceId});
  }

  void finalize() override { getData().WasFinalized = true; }
};

static SummaryDataBuilderRegistry::Add<Analysis1Builder>
    RegAnalysis1("Builder for Analysis1");

class Analysis2Builder final
    : public SummaryDataBuilder<Analysis2Data, Analysis2EntitySummary> {
public:
  ~Analysis2Builder() { Analysis2BuilderWasDestroyed = true; }

  void addSummary(EntityId Id,
                  std::unique_ptr<Analysis2EntitySummary> S) override {
    getData().Entries.push_back({Id, S->InstanceId});
  }

  void finalize() override { getData().WasFinalized = true; }
};

static SummaryDataBuilderRegistry::Add<Analysis2Builder>
    RegAnalysis2("Builder for Analysis2");

class Analysis4Builder final
    : public SummaryDataBuilder<Analysis4Data, Analysis4EntitySummary> {
public:
  ~Analysis4Builder() { Analysis4BuilderWasDestroyed = true; }

  void addSummary(EntityId Id,
                  std::unique_ptr<Analysis4EntitySummary> S) override {
    getData().Entries.push_back({Id, S->InstanceId});
  }

  void finalize() override { getData().WasFinalized = true; }
};

static SummaryDataBuilderRegistry::Add<Analysis4Builder>
    RegAnalysis4("Builder for Analysis4");

// ---------------------------------------------------------------------------
// Fixture
// ---------------------------------------------------------------------------

class LUSummaryConsumerTest : public TestFixture {
protected:
  static constexpr EntityLinkage ExternalLinkage =
      EntityLinkage(EntityLinkageType::External);

  void SetUp() override {
    NextSummaryInstanceId = 0;
    Analysis1BuilderWasDestroyed = false;
    Analysis2BuilderWasDestroyed = false;
    Analysis4BuilderWasDestroyed = false;
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

  /// Inserts a freshly constructed SummaryT for the given entity and returns
  /// the summary's InstanceId so the test can verify delivery later.
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

TEST(SummaryDataBuilderRegistryTest, BuilderIsRegistered) {
  EXPECT_FALSE(SummaryDataBuilderRegistry::contains("AnalysisNonExisting"));
  EXPECT_TRUE(SummaryDataBuilderRegistry::contains("Analysis1"));
  EXPECT_TRUE(SummaryDataBuilderRegistry::contains("Analysis2"));
  EXPECT_TRUE(SummaryDataBuilderRegistry::contains("Analysis4"));
}

TEST(SummaryDataBuilderRegistryTest, BuilderCanBeInstantiated) {
  EXPECT_EQ(SummaryDataBuilderRegistry::instantiate("AnalysisNonExisting"),
            nullptr);
  EXPECT_NE(SummaryDataBuilderRegistry::instantiate("Analysis1"), nullptr);
  EXPECT_NE(SummaryDataBuilderRegistry::instantiate("Analysis2"), nullptr);
  EXPECT_NE(SummaryDataBuilderRegistry::instantiate("Analysis4"), nullptr);
}

// run() — processes all registered analyses present in the LUSummary.
TEST_F(LUSummaryConsumerTest, RunAll) {
  auto LU = makeLUSummary();
  const auto E1 = addEntity(*LU, "Entity1");
  const auto E2 = addEntity(*LU, "Entity2");
  const auto E3 = addEntity(*LU, "Entity3");

  int s1a = insertSummary<Analysis1EntitySummary>(*LU, "Analysis1", E1);
  int s1b = insertSummary<Analysis1EntitySummary>(*LU, "Analysis1", E2);
  int s2a = insertSummary<Analysis2EntitySummary>(*LU, "Analysis2", E2);
  int s2b = insertSummary<Analysis2EntitySummary>(*LU, "Analysis2", E3);

  // No registered builder — Analysis3 data silently skipped.
  (void)insertSummary<Analysis3EntitySummary>(*LU, "Analysis3", E1);

  LUSummaryConsumer Consumer(std::move(LU));
  SummaryDataStore Store = std::move(Consumer).run();

  {
    auto Data1OrErr = Store.get<Analysis1Data>();
    ASSERT_THAT_EXPECTED(Data1OrErr, llvm::Succeeded());
    auto &Data1 = *Data1OrErr;

    EXPECT_EQ(Data1.Entries.size(), 2u);
    EXPECT_TRUE(hasEntry(Data1.Entries, E1, s1a));
    EXPECT_TRUE(hasEntry(Data1.Entries, E2, s1b));
    EXPECT_TRUE(Data1.WasFinalized);
    EXPECT_TRUE(Analysis1BuilderWasDestroyed);

    // take() transfers ownership — subsequent get() returns an error.
    auto Take1OrErr = Store.take<Analysis1Data>();
    ASSERT_THAT_EXPECTED(Take1OrErr, llvm::Succeeded());
    EXPECT_NE(*Take1OrErr, nullptr);
    EXPECT_THAT_EXPECTED(Store.get<Analysis1Data>(), llvm::Failed());
  }

  {
    auto Data2OrErr = Store.get<Analysis2Data>();
    ASSERT_THAT_EXPECTED(Data2OrErr, llvm::Succeeded());
    auto &Data2 = *Data2OrErr;
    EXPECT_EQ(Data2.Entries.size(), 2u);
    EXPECT_TRUE(hasEntry(Data2.Entries, E2, s2a));
    EXPECT_TRUE(hasEntry(Data2.Entries, E3, s2b));
    EXPECT_TRUE(Data2.WasFinalized);
    EXPECT_TRUE(Analysis2BuilderWasDestroyed);

    auto Take2OrErr = Store.take<Analysis2Data>();
    ASSERT_THAT_EXPECTED(Take2OrErr, llvm::Succeeded());
    EXPECT_NE(*Take2OrErr, nullptr);
    EXPECT_THAT_EXPECTED(Store.get<Analysis2Data>(), llvm::Failed());
  }

  // Unregistered — not present in store.
  EXPECT_THAT_EXPECTED(Store.get<Analysis3Data>(), llvm::Failed());

  // Registered builder but no data in LUSummary — not present in store.
  EXPECT_THAT_EXPECTED(Store.get<Analysis4Data>(), llvm::Failed());
  EXPECT_FALSE(Analysis4BuilderWasDestroyed);
}

// run(names) — processes only the analyses for the given names.
TEST_F(LUSummaryConsumerTest, RunByName) {
  auto LU = makeLUSummary();
  const auto E1 = addEntity(*LU, "Entity1");
  const auto E2 = addEntity(*LU, "Entity2");

  int s1a = insertSummary<Analysis1EntitySummary>(*LU, "Analysis1", E1);
  insertSummary<Analysis2EntitySummary>(*LU, "Analysis2", E2);

  LUSummaryConsumer Consumer(std::move(LU));
  auto StoreOrErr = Consumer.run({SummaryName("Analysis1")});
  ASSERT_THAT_EXPECTED(StoreOrErr, llvm::Succeeded());

  // Analysis1 was requested and has data — present.
  auto Data1OrErr = StoreOrErr->get<Analysis1Data>();
  ASSERT_THAT_EXPECTED(Data1OrErr, llvm::Succeeded());
  auto &Data1 = *Data1OrErr;
  EXPECT_EQ(Data1.Entries.size(), 1u);
  EXPECT_TRUE(hasEntry(Data1.Entries, E1, s1a));
  EXPECT_TRUE(Data1.WasFinalized);

  // Analysis2 was not requested — not present even though data exists.
  EXPECT_THAT_EXPECTED(StoreOrErr->get<Analysis2Data>(), llvm::Failed());
}

// run(names) — error when a requested name has no data in LUSummary.
TEST_F(LUSummaryConsumerTest, RunByNameErrorMissingData) {
  auto LU = makeLUSummary();
  LUSummaryConsumer Consumer(std::move(LU));

  EXPECT_THAT_EXPECTED(Consumer.run({SummaryName("Analysis1")}),
                       llvm::Failed());
}

// run(names) — error when a requested name has no registered builder.
TEST_F(LUSummaryConsumerTest, RunByNameErrorMissingBuilder) {
  auto LU = makeLUSummary();
  const auto E1 = addEntity(*LU, "Entity1");
  insertSummary<Analysis3EntitySummary>(*LU, "Analysis3", E1);

  LUSummaryConsumer Consumer(std::move(LU));

  // Analysis3 has data but no registered builder.
  EXPECT_THAT_EXPECTED(Consumer.run({SummaryName("Analysis3")}),
                       llvm::Failed());
}

// run<DataTs...>() — type-safe subset.
TEST_F(LUSummaryConsumerTest, RunByType) {
  auto LU = makeLUSummary();
  const auto E1 = addEntity(*LU, "Entity1");
  const auto E2 = addEntity(*LU, "Entity2");

  int s1a = insertSummary<Analysis1EntitySummary>(*LU, "Analysis1", E1);
  insertSummary<Analysis2EntitySummary>(*LU, "Analysis2", E2);

  LUSummaryConsumer Consumer(std::move(LU));
  auto StoreOrErr = Consumer.run<Analysis1Data>();
  ASSERT_THAT_EXPECTED(StoreOrErr, llvm::Succeeded());

  // Analysis1 was requested — present.
  auto Data1OrErr = StoreOrErr->get<Analysis1Data>();
  ASSERT_THAT_EXPECTED(Data1OrErr, llvm::Succeeded());
  auto &Data1 = *Data1OrErr;
  EXPECT_EQ(Data1.Entries.size(), 1u);
  EXPECT_TRUE(hasEntry(Data1.Entries, E1, s1a));
  EXPECT_TRUE(Data1.WasFinalized);

  // Analysis2 was not requested — not present even though data exists.
  EXPECT_THAT_EXPECTED(StoreOrErr->get<Analysis2Data>(), llvm::Failed());
}

// run<DataTs...>() — error when a requested type has no data in LUSummary.
TEST_F(LUSummaryConsumerTest, RunByTypeErrorMissingData) {
  auto LU = makeLUSummary();
  LUSummaryConsumer Consumer(std::move(LU));

  EXPECT_THAT_EXPECTED(Consumer.run<Analysis1Data>(), llvm::Failed());
}

// contains() — present entries return true; absent entries return false.
TEST_F(LUSummaryConsumerTest, Contains) {
  auto LU = makeLUSummary();
  const auto E1 = addEntity(*LU, "Entity1");
  insertSummary<Analysis1EntitySummary>(*LU, "Analysis1", E1);

  LUSummaryConsumer Consumer(std::move(LU));
  SummaryDataStore Store = std::move(Consumer).run();

  // Type-safe variant.
  EXPECT_TRUE(Store.contains<Analysis1Data>());
  EXPECT_FALSE(Store.contains<Analysis2Data>());

  // Name-based variant.
  EXPECT_TRUE(Store.contains(SummaryName("Analysis1")));
  EXPECT_FALSE(Store.contains(SummaryName("Analysis2")));

  // After take(), contains() returns false.
  ASSERT_THAT_EXPECTED(Store.take<Analysis1Data>(), llvm::Succeeded());
  EXPECT_FALSE(Store.contains<Analysis1Data>());
  EXPECT_FALSE(Store.contains(SummaryName("Analysis1")));
}

} // namespace
