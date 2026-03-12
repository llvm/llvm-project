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
#include "gtest/gtest.h"
#include <algorithm>
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

class Analysis1EntitySummary : public EntitySummary {
public:
  int InstanceId = NextSummaryInstanceId++;
  SummaryName getSummaryName() const override {
    return SummaryName("Analysis1");
  }
};

class Analysis2EntitySummary : public EntitySummary {
public:
  int InstanceId = NextSummaryInstanceId++;
  SummaryName getSummaryName() const override {
    return SummaryName("Analysis2");
  }
};

class Analysis4EntitySummary : public EntitySummary {
public:
  int InstanceId = NextSummaryInstanceId++;
  SummaryName getSummaryName() const override {
    return SummaryName("Analysis4");
  }
};

// ---------------------------------------------------------------------------
// Data
// ---------------------------------------------------------------------------

class Analysis1Data : public SummaryData {
public:
  static SummaryName summaryName() { return SummaryName("Analysis1"); }
  std::vector<std::pair<EntityId, int>> Entries;
  bool WasFinalized = false;
};

class Analysis2Data : public SummaryData {
public:
  static SummaryName summaryName() { return SummaryName("Analysis2"); }
  std::vector<std::pair<EntityId, int>> Entries;
  bool WasFinalized = false;
};

// No builder or registration for Analysis3. Data for Analysis3 is inserted
// into the LUSummary to verify the consumer silently skips it.
class Analysis3Data : public SummaryData {
public:
  static SummaryName summaryName() { return SummaryName("Analysis3"); }
};

// Analysis4 has a registered builder but no data is inserted into the
// LUSummary, so the builder is never invoked and getData returns nullptr.
class Analysis4Data : public SummaryData {
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

class Analysis1Builder
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

class Analysis2Builder
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

class Analysis4Builder
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
    return std::find(Entries.begin(), Entries.end(),
                     std::make_pair(Id, InstanceId)) != Entries.end();
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
  EXPECT_FALSE(SummaryDataBuilderRegistry::isRegistered("Analysis10"));
  EXPECT_TRUE(SummaryDataBuilderRegistry::isRegistered("Analysis1"));
  EXPECT_TRUE(SummaryDataBuilderRegistry::isRegistered("Analysis2"));
  EXPECT_TRUE(SummaryDataBuilderRegistry::isRegistered("Analysis4"));
}

TEST(SummaryDataBuilderRegistryTest, BuilderCanBeInstantiated) {
  EXPECT_EQ(SummaryDataBuilderRegistry::instantiate("Analysis10"), nullptr);
  EXPECT_NE(SummaryDataBuilderRegistry::instantiate("Analysis1"), nullptr);
  EXPECT_NE(SummaryDataBuilderRegistry::instantiate("Analysis2"), nullptr);
  EXPECT_NE(SummaryDataBuilderRegistry::instantiate("Analysis4"), nullptr);
}

TEST_F(LUSummaryConsumerTest, Run) {
  auto LU = makeLUSummary();
  const auto E1 = addEntity(*LU, "Entity1");
  const auto E2 = addEntity(*LU, "Entity2");
  const auto E3 = addEntity(*LU, "Entity3");

  int s1a = insertSummary<Analysis1EntitySummary>(*LU, "Analysis1", E1);
  int s1b = insertSummary<Analysis1EntitySummary>(*LU, "Analysis1", E2);
  int s2a = insertSummary<Analysis2EntitySummary>(*LU, "Analysis2", E2);
  int s2b = insertSummary<Analysis2EntitySummary>(*LU, "Analysis2", E3);

  // no registered builder
  [[maybe_unused]] int s3a =
      insertSummary<Analysis1EntitySummary>(*LU, "Analysis3", E1);

  LUSummaryConsumer Consumer(std::move(LU));
  Consumer.run();

  // Analysis1
  {
    auto Data1 = Consumer.getData<Analysis1Data>();
    ASSERT_NE(Data1, nullptr);

    // getData ownership transfer — second call returns nullptr
    EXPECT_EQ(Consumer.getData<Analysis1Data>(), nullptr);

    EXPECT_EQ(Data1->Entries.size(), 2u);
    EXPECT_TRUE(hasEntry(Data1->Entries, E1, s1a));
    EXPECT_TRUE(hasEntry(Data1->Entries, E2, s1b));

    EXPECT_TRUE(Data1->WasFinalized);

    // Builder lifetime
    EXPECT_TRUE(Analysis1BuilderWasDestroyed);
  }

  // Analysis2
  {
    auto Data2 = Consumer.getData<Analysis2Data>();
    ASSERT_NE(Data2, nullptr);

    // getData ownership transfer — second call returns nullptr
    EXPECT_EQ(Consumer.getData<Analysis2Data>(), nullptr);

    EXPECT_EQ(Data2->Entries.size(), 2u);
    EXPECT_TRUE(hasEntry(Data2->Entries, E2, s2a));
    EXPECT_TRUE(hasEntry(Data2->Entries, E3, s2b));

    EXPECT_TRUE(Data2->WasFinalized);

    // Builder lifetime
    EXPECT_TRUE(Analysis2BuilderWasDestroyed);
  }

  // Analysis3
  {
    // Unregistered builder — Analysis3 data silently skipped
    EXPECT_EQ(Consumer.getData<Analysis3Data>(), nullptr);
  }

  // Analysis4
  {
    // Registered builder but no data in LUSummary — builder never invoked
    EXPECT_EQ(Consumer.getData<Analysis4Data>(), nullptr);

    // Builder lifetime
    EXPECT_FALSE(Analysis4BuilderWasDestroyed);
  }
}

} // namespace
