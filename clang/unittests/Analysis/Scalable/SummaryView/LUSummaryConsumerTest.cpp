//===- LUSummaryConsumerTest.cpp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/Scalable/SummaryView/LUSummaryConsumer.h"
#include "../TestFixture.h"
#include "MockSummaryViewBuilders.h"
#include "clang/Analysis/Scalable/EntityLinker/LUSummary.h"
#include "clang/Analysis/Scalable/Model/BuildNamespace.h"
#include "clang/Analysis/Scalable/Model/EntityId.h"
#include "clang/Analysis/Scalable/Model/EntityIdTable.h"
#include "clang/Analysis/Scalable/Model/EntityName.h"
#include "clang/Analysis/Scalable/Model/SummaryName.h"
#include "gtest/gtest.h"
#include <memory>

using namespace clang;
using namespace ssaf;

namespace {

class LUSummaryConsumerTest : public TestFixture {
protected:
  void SetUp() override { clearMockBuilderLog(); }

  std::unique_ptr<LUSummary> makeLUSummary() {
    NestedBuildNamespace NS(
        {BuildNamespace(BuildNamespaceKind::LinkUnit, "TestLU")});
    return std::make_unique<LUSummary>(std::move(NS));
  }

  /// Add an entity to the LUSummary's id table and return its EntityId.
  EntityId addEntity(LUSummary &LU, llvm::StringRef USR) {
    NestedBuildNamespace NS(
        {BuildNamespace(BuildNamespaceKind::LinkUnit, "TestLU")});
    EntityName Name(USR.str(), "", NS);
    return getIdTable(LU).getId(Name);
  }

  /// Insert a pre-built EntitySummary into LUSummary::Data.
  void insertSummary(LUSummary &LU, llvm::StringRef SummaryNameStr, EntityId Id,
                     std::unique_ptr<EntitySummary> Summary) {
    getData(LU)[SummaryName(SummaryNameStr.str())][Id] = std::move(Summary);
  }
};

// ---------------------------------------------------------------------------
// No matching data in LUSummary
// ---------------------------------------------------------------------------

TEST_F(LUSummaryConsumerTest, NoMatchingData_AddSummaryNeverCalled) {
  auto LU = makeLUSummary();
  // Insert data for a summary name that has no registered builder.
  auto Id = addEntity(*LU, "A");
  insertSummary(*LU, "UnregisteredAnalysis", Id,
                std::make_unique<MockEntitySummary1>());

  LUSummaryConsumer Consumer(std::move(LU));
  Consumer.run();

  EXPECT_EQ(MockBuilderLog.find("addSummary"), std::string::npos);
}

TEST_F(LUSummaryConsumerTest, NoMatchingData_FinalizeNotCalled) {
  auto LU = makeLUSummary(); // empty — no Mock1 data

  LUSummaryConsumer Consumer(std::move(LU));
  Consumer.run();

  EXPECT_EQ(MockBuilderLog.find("finalize"), std::string::npos);
}

TEST_F(LUSummaryConsumerTest, NoMatchingData_GetViewReturnsNull) {
  auto LU = makeLUSummary();

  LUSummaryConsumer Consumer(std::move(LU));
  Consumer.run();

  EXPECT_EQ(Consumer.getView<MockView1>(), nullptr);
}

// ---------------------------------------------------------------------------
// Matching data delivered correctly
// ---------------------------------------------------------------------------

TEST_F(LUSummaryConsumerTest, MatchingData_AddSummaryCalledPerEntity) {
  auto LU = makeLUSummary();
  auto Id1 = addEntity(*LU, "A");
  auto Id2 = addEntity(*LU, "B");
  insertSummary(*LU, "Mock1", Id1, std::make_unique<MockEntitySummary1>());
  insertSummary(*LU, "Mock1", Id2, std::make_unique<MockEntitySummary1>());

  LUSummaryConsumer Consumer(std::move(LU));
  Consumer.run();

  auto View = Consumer.getView<MockView1>();
  ASSERT_NE(View, nullptr);
  EXPECT_EQ(View->Ids.size(), 2u);
  EXPECT_NE(std::find(View->Ids.begin(), View->Ids.end(), Id1),
            View->Ids.end());
  EXPECT_NE(std::find(View->Ids.begin(), View->Ids.end(), Id2),
            View->Ids.end());
}

TEST_F(LUSummaryConsumerTest, MatchingData_FinalizeCalledAfterAllAddSummary) {
  auto LU = makeLUSummary();
  auto Id = addEntity(*LU, "A");
  insertSummary(*LU, "Mock1", Id, std::make_unique<MockEntitySummary1>());

  LUSummaryConsumer Consumer(std::move(LU));
  Consumer.run();

  // Verify ordering in the log: all addSummary entries must precede all
  // finalize entries.
  auto AddSummaryPos = MockBuilderLog.find("addSummary");
  auto FinalizePos = MockBuilderLog.find("finalize");
  ASSERT_NE(AddSummaryPos, std::string::npos);
  ASSERT_NE(FinalizePos, std::string::npos);
  EXPECT_LT(AddSummaryPos, FinalizePos);
}

// ---------------------------------------------------------------------------
// Multiple builders receive only their own entities
// ---------------------------------------------------------------------------

TEST_F(LUSummaryConsumerTest, MultipleBuilders_IndependentEntities) {
  auto LU = makeLUSummary();
  auto Id1 = addEntity(*LU, "A");
  auto Id2 = addEntity(*LU, "B");
  insertSummary(*LU, "Mock1", Id1, std::make_unique<MockEntitySummary1>());
  insertSummary(*LU, "Mock2", Id2, std::make_unique<MockEntitySummary2>());

  LUSummaryConsumer Consumer(std::move(LU));
  Consumer.run();

  auto View1 = Consumer.getView<MockView1>();
  ASSERT_NE(View1, nullptr);
  ASSERT_EQ(View1->Ids.size(), 1u);
  EXPECT_EQ(View1->Ids[0], Id1);

  auto View2 = Consumer.getView<MockView2>();
  ASSERT_NE(View2, nullptr);
  ASSERT_EQ(View2->Ids.size(), 1u);
  EXPECT_EQ(View2->Ids[0], Id2);
}

// ---------------------------------------------------------------------------
// getView ownership transfer
// ---------------------------------------------------------------------------

TEST_F(LUSummaryConsumerTest, GetView_FirstCallReturnsNonNull) {
  auto LU = makeLUSummary();
  auto Id = addEntity(*LU, "A");
  insertSummary(*LU, "Mock1", Id, std::make_unique<MockEntitySummary1>());

  LUSummaryConsumer Consumer(std::move(LU));
  Consumer.run();

  EXPECT_NE(Consumer.getView<MockView1>(), nullptr);
}

TEST_F(LUSummaryConsumerTest, GetView_SecondCallReturnsNull) {
  auto LU = makeLUSummary();
  auto Id = addEntity(*LU, "A");
  insertSummary(*LU, "Mock1", Id, std::make_unique<MockEntitySummary1>());

  LUSummaryConsumer Consumer(std::move(LU));
  Consumer.run();

  auto First = Consumer.getView<MockView1>();
  EXPECT_NE(First, nullptr);
  EXPECT_EQ(Consumer.getView<MockView1>(), nullptr);
}

TEST_F(LUSummaryConsumerTest, GetView_UnregisteredViewReturnsNull) {
  // MockView1 is registered; a hypothetical OtherView is not.
  struct OtherView : public SummaryView {
    static SummaryName summaryName() { return SummaryName("Other"); }
  };

  auto LU = makeLUSummary();
  LUSummaryConsumer Consumer(std::move(LU));
  Consumer.run();

  EXPECT_EQ(Consumer.getView<OtherView>(), nullptr);
}

// ---------------------------------------------------------------------------
// Builder lifetime
// ---------------------------------------------------------------------------

TEST_F(LUSummaryConsumerTest, BuildersDestroyedAfterRun) {
  auto LU = makeLUSummary();
  auto Id1 = addEntity(*LU, "A");
  auto Id2 = addEntity(*LU, "B");
  insertSummary(*LU, "Mock1", Id1, std::make_unique<MockEntitySummary1>());
  insertSummary(*LU, "Mock2", Id2, std::make_unique<MockEntitySummary2>());

  LUSummaryConsumer Consumer(std::move(LU));
  Consumer.run();

  // Both destructors must have been called by the time run() returns.
  EXPECT_NE(
      MockBuilderLog.find("MockSummaryViewBuilder1 destructor was invoked"),
      std::string::npos);
  EXPECT_NE(
      MockBuilderLog.find("MockSummaryViewBuilder2 destructor was invoked"),
      std::string::npos);
}

} // namespace
