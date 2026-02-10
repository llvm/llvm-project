//===- unittests/Analysis/Scalable/TUSummaryBuilderTest.cpp ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/Scalable/TUSummary/TUSummaryBuilder.h"
#include "TestFixture.h"
#include "clang/Analysis/Scalable/Model/BuildNamespace.h"
#include "clang/Analysis/Scalable/Model/EntityId.h"
#include "clang/Analysis/Scalable/Model/EntityName.h"
#include "clang/Analysis/Scalable/Model/SummaryName.h"
#include "clang/Analysis/Scalable/Serialization/SerializationFormat.h"
#include "clang/Analysis/Scalable/TUSummary/EntitySummary.h"
#include "clang/Analysis/Scalable/TUSummary/TUSummary.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <type_traits>

using namespace clang;
using namespace ssaf;

using llvm::SmallVector;
using testing::Field;
using testing::Optional;
using testing::UnorderedElementsAre;

[[nodiscard]]
static TUSummary makeFakeSummary() {
  BuildNamespace NS(BuildNamespaceKind::CompilationUnit, "Mock.cpp");
  TUSummary Summary(NS);
  return Summary;
}

[[nodiscard]]
static EntityId addTestEntity(TUSummaryBuilder &Builder, llvm::StringRef USR) {
  EntityName EN(USR, "", /*Namespace=*/{});
  EntityLinkage MockLinkage;
  return Builder.addEntity(EN, MockLinkage);
}

template <class ConcreteEntitySummary>
[[nodiscard]]
static SummaryName addFactTo(TUSummaryBuilder &Builder, EntityId ID,
                             ConcreteEntitySummary Fact) {
  static_assert(std::is_base_of_v<EntitySummary, ConcreteEntitySummary>);
  auto NewFact = std::make_unique<ConcreteEntitySummary>(std::move(Fact));
  SummaryName Name = NewFact->getSummaryName();
  Builder.addFact(ID, std::move(NewFact));
  return Name;
}

namespace {

// Mock EntitySummary classes for testing
struct MockSummaryData1 final
    : public llvm::RTTIExtends<MockSummaryData1, EntitySummary> {
  explicit MockSummaryData1(int Value) : Value(Value) {}
  SummaryName getSummaryName() const override {
    return SummaryName("MockSummary1");
  }
  int Value = 0;
  static char ID;
};

struct MockSummaryData2 final
    : public llvm::RTTIExtends<MockSummaryData2, EntitySummary> {
  explicit MockSummaryData2(std::string Text) : Text(std::move(Text)) {}
  SummaryName getSummaryName() const override {
    return SummaryName("MockSummary2");
  }
  std::string Text;
  static char ID;
};

struct MockSummaryData3 final
    : public llvm::RTTIExtends<MockSummaryData3, EntitySummary> {
  explicit MockSummaryData3(bool Flag) : Flag(Flag) {}
  SummaryName getSummaryName() const override {
    return SummaryName("MockSummary3");
  }
  bool Flag = false;
  static char ID;
};

char MockSummaryData1::ID = 0;
char MockSummaryData2::ID = 0;
char MockSummaryData3::ID = 0;

void PrintTo(const MockSummaryData1 &S, std::ostream *OS) {
  *OS << "MockSummaryData1(" << S.getSummaryName().str().str() << ")";
}
void PrintTo(const MockSummaryData2 &S, std::ostream *OS) {
  *OS << "MockSummaryData2(" << S.getSummaryName().str().str() << ")";
}
void PrintTo(const MockSummaryData3 &S, std::ostream *OS) {
  *OS << "MockSummaryData3(" << S.getSummaryName().str().str() << ")";
}

struct TUSummaryBuilderTest : ssaf::TestFixture {
  static llvm::SmallVector<SummaryName> summaryNames(const TUSummary &Summary) {
    return llvm::to_vector(llvm::make_first_range(getData(Summary)));
  }

  static llvm::SmallVector<EntityId>
  entitiesOfSummary(const TUSummary &Summary, const SummaryName &Name) {
    const auto &MappingIt = getData(Summary).find(Name);
    if (MappingIt == getData(Summary).end())
      return {};
    return llvm::to_vector(llvm::make_first_range(MappingIt->second));
  }

  template <class ConcreteSummaryData>
  static std::optional<ConcreteSummaryData>
  getAsEntitySummary(const TUSummary &Summary, const SummaryName &Name,
                     EntityId E) {
    static_assert(std::is_base_of_v<EntitySummary, ConcreteSummaryData>);
    const auto &MappingIt = getData(Summary).find(Name);
    if (MappingIt == getData(Summary).end())
      return std::nullopt;
    auto SummaryIt = MappingIt->second.find(E);
    if (SummaryIt == MappingIt->second.end())
      return std::nullopt;
    assert(llvm::isa<ConcreteSummaryData>(*SummaryIt->second));
    return llvm::cast<ConcreteSummaryData>(*SummaryIt->second);
  }
};

TEST_F(TUSummaryBuilderTest, AddEntity) {
  TUSummary Summary = makeFakeSummary();
  TUSummaryBuilder Builder(Summary);

  EntityName EN1("c:@F@foo", "", /*Namespace=*/{});
  EntityName EN2("c:@F@bar", "", /*Namespace=*/{});

  EntityLinkage MockLinkage;
  EntityId ID = Builder.addEntity(EN1, MockLinkage);
  EntityId IDAlias = Builder.addEntity(EN1, MockLinkage);
  EXPECT_EQ(ID, IDAlias); // Idenpotency

  EntityId ID2 = Builder.addEntity(EN2, MockLinkage);
  EXPECT_NE(ID, ID2);
  EXPECT_NE(IDAlias, ID2);

  const EntityIdTable &IdTable = getIdTable(Summary);
  EXPECT_EQ(IdTable.count(), 2U);
  EXPECT_TRUE(IdTable.contains(EN1));
  EXPECT_TRUE(IdTable.contains(EN2));

  const auto &Entities = getEntities(Summary);
  EXPECT_EQ(Entities.size(), 2U);
  ASSERT_EQ(Entities.count(ID), 1U);
  EXPECT_EQ(Entities.find(ID)->second, MockLinkage);

  ASSERT_EQ(Entities.count(ID2), 1U);
  EXPECT_EQ(Entities.find(ID2)->second, MockLinkage);
}

TEST_F(TUSummaryBuilderTest, TUSummaryBuilderAddSingleFact) {
  TUSummary Summary = makeFakeSummary();
  TUSummaryBuilder Builder(Summary);

  EntityId ID = addTestEntity(Builder, "c:@F@foo");
  SummaryName Name = addFactTo(Builder, ID, MockSummaryData1(10));

  // Should have a summary type with an entity.
  EXPECT_THAT(summaryNames(Summary), UnorderedElementsAre(Name));
  EXPECT_THAT(entitiesOfSummary(Summary, Name), UnorderedElementsAre(ID));

  EXPECT_THAT(getAsEntitySummary<MockSummaryData1>(Summary, Name, ID),
              Optional(Field(&MockSummaryData1::Value, 10)));
}

TEST_F(TUSummaryBuilderTest, AddMultipleFactsToSameEntity) {
  TUSummary Summary = makeFakeSummary();
  TUSummaryBuilder Builder(Summary);
  EntityId ID = addTestEntity(Builder, "c:@F@foo");

  // Add different summary types to the same entity.
  SummaryName Name1 = addFactTo(Builder, ID, MockSummaryData1(42));
  SummaryName Name2 = addFactTo(Builder, ID, MockSummaryData2("test data"));
  SummaryName Name3 = addFactTo(Builder, ID, MockSummaryData3(true));

  // All Names must be unique
  EXPECT_EQ((std::set<SummaryName>{Name1, Name2, Name3}.size()), 3U);

  // Should have 3 summary type with the same entity.
  EXPECT_THAT(summaryNames(Summary), UnorderedElementsAre(Name1, Name2, Name3));
  EXPECT_THAT(entitiesOfSummary(Summary, Name1), UnorderedElementsAre(ID));
  EXPECT_THAT(entitiesOfSummary(Summary, Name2), UnorderedElementsAre(ID));
  EXPECT_THAT(entitiesOfSummary(Summary, Name3), UnorderedElementsAre(ID));

  EXPECT_THAT(getAsEntitySummary<MockSummaryData1>(Summary, Name1, ID),
              Optional(Field(&MockSummaryData1::Value, 42)));
  EXPECT_THAT(getAsEntitySummary<MockSummaryData2>(Summary, Name2, ID),
              Optional(Field(&MockSummaryData2::Text, "test data")));
  EXPECT_THAT(getAsEntitySummary<MockSummaryData3>(Summary, Name3, ID),
              Optional(Field(&MockSummaryData3::Flag, true)));
}

TEST_F(TUSummaryBuilderTest, AddSameFactTypeToMultipleEntities) {
  TUSummary Summary = makeFakeSummary();
  TUSummaryBuilder Builder(Summary);

  EntityId ID1 = addTestEntity(Builder, "c:@F@foo");
  EntityId ID2 = addTestEntity(Builder, "c:@F@bar");
  EntityId ID3 = addTestEntity(Builder, "c:@F@baz");

  // Add the same summary type to different entities.
  SummaryName Name1 = addFactTo(Builder, ID1, MockSummaryData1(1));
  SummaryName Name2 = addFactTo(Builder, ID2, MockSummaryData1(2));
  SummaryName Name3 = addFactTo(Builder, ID3, MockSummaryData1(3));

  // All 3 should be the same summary type.
  EXPECT_THAT((llvm::ArrayRef{Name1, Name2, Name3}), testing::Each(Name1));

  // Should have only 1 summary type with 3 entities.
  EXPECT_THAT(summaryNames(Summary), UnorderedElementsAre(Name1));
  EXPECT_THAT(entitiesOfSummary(Summary, Name1),
              UnorderedElementsAre(ID1, ID2, ID3));

  EXPECT_THAT(getAsEntitySummary<MockSummaryData1>(Summary, Name1, ID1),
              Optional(Field(&MockSummaryData1::Value, 1)));
  EXPECT_THAT(getAsEntitySummary<MockSummaryData1>(Summary, Name2, ID2),
              Optional(Field(&MockSummaryData1::Value, 2)));
  EXPECT_THAT(getAsEntitySummary<MockSummaryData1>(Summary, Name3, ID3),
              Optional(Field(&MockSummaryData1::Value, 3)));
}

TEST_F(TUSummaryBuilderTest, AddFactReplacesExistingFact) {
  TUSummary Summary = makeFakeSummary();
  TUSummaryBuilder Builder(Summary);
  EntityId ID = addTestEntity(Builder, "c:@F@foo");

  SummaryName Name = addFactTo(Builder, ID, MockSummaryData1(10));

  // Check the initial value.
  EXPECT_THAT(summaryNames(Summary), UnorderedElementsAre(Name));
  EXPECT_THAT(entitiesOfSummary(Summary, Name), UnorderedElementsAre(ID));
  EXPECT_THAT(getAsEntitySummary<MockSummaryData1>(Summary, Name, ID),
              Optional(Field(&MockSummaryData1::Value, 10)));

  // Add another fact with the same SummaryName.
  // This should replace the previous fact.
  SummaryName ReplacementName = addFactTo(Builder, ID, MockSummaryData1(20));
  ASSERT_EQ(ReplacementName, Name);

  // Check that the value was replaced.
  EXPECT_THAT(summaryNames(Summary), UnorderedElementsAre(Name));
  EXPECT_THAT(entitiesOfSummary(Summary, Name), UnorderedElementsAre(ID));
  EXPECT_THAT(getAsEntitySummary<MockSummaryData1>(Summary, Name, ID),
              Optional(Field(&MockSummaryData1::Value, 20)));
}

TEST_F(TUSummaryBuilderTest, AddFactsComplexScenario) {
  TUSummary Summary = makeFakeSummary();
  TUSummaryBuilder Builder(Summary);

  EntityId ID1 = addTestEntity(Builder, "c:@F@foo");
  EntityId ID2 = addTestEntity(Builder, "c:@F@bar");

  SummaryName Name1 = addFactTo(Builder, ID1, MockSummaryData1(10));
  SummaryName Name2 = addFactTo(Builder, ID1, MockSummaryData2("twenty"));

  SummaryName Name3 = addFactTo(Builder, ID2, MockSummaryData1(30));
  SummaryName Name4 = addFactTo(Builder, ID2, MockSummaryData3(true));

  // Check that we have only 3 distinct summary names.
  EXPECT_EQ(Name1, Name3);
  EXPECT_THAT((std::set{Name1, Name2, Name3, Name4}),
              UnorderedElementsAre(Name1, Name2, Name4));

  // Check that we have two facts for the two summaries each.
  EXPECT_THAT(summaryNames(Summary), UnorderedElementsAre(Name1, Name2, Name4));
  EXPECT_THAT(entitiesOfSummary(Summary, Name1),
              UnorderedElementsAre(ID1, ID2));
  EXPECT_THAT(entitiesOfSummary(Summary, Name2), UnorderedElementsAre(ID1));
  EXPECT_THAT(entitiesOfSummary(Summary, Name4), UnorderedElementsAre(ID2));

  EXPECT_THAT(getAsEntitySummary<MockSummaryData1>(Summary, Name1, ID1),
              Optional(Field(&MockSummaryData1::Value, 10)));
  EXPECT_THAT(getAsEntitySummary<MockSummaryData1>(Summary, Name1, ID2),
              Optional(Field(&MockSummaryData1::Value, 30)));
  EXPECT_THAT(getAsEntitySummary<MockSummaryData2>(Summary, Name2, ID1),
              Optional(Field(&MockSummaryData2::Text, "twenty")));
  EXPECT_THAT(getAsEntitySummary<MockSummaryData3>(Summary, Name4, ID2),
              Optional(Field(&MockSummaryData3::Flag, true)));

  // Replace a fact of Name1 on entity 1 with the new value 50.
  SummaryName Name5 = addFactTo(Builder, ID1, MockSummaryData1(50));
  ASSERT_EQ(Name5, Name1);

  // Check that the summary names and entity IDs didn't change.
  EXPECT_THAT(summaryNames(Summary), UnorderedElementsAre(Name1, Name2, Name4));
  EXPECT_THAT(entitiesOfSummary(Summary, Name1),
              UnorderedElementsAre(ID1, ID2));
  EXPECT_THAT(entitiesOfSummary(Summary, Name2), UnorderedElementsAre(ID1));
  EXPECT_THAT(entitiesOfSummary(Summary, Name4), UnorderedElementsAre(ID2));

  // Check the Name1 ID1 entity summary value was changed to 50.
  EXPECT_THAT(getAsEntitySummary<MockSummaryData1>(Summary, Name1, ID1),
              Optional(Field(&MockSummaryData1::Value, 50)));

  // Check that the rest remained the same.
  EXPECT_THAT(getAsEntitySummary<MockSummaryData1>(Summary, Name1, ID2),
              Optional(Field(&MockSummaryData1::Value, 30)));
  EXPECT_THAT(getAsEntitySummary<MockSummaryData2>(Summary, Name2, ID1),
              Optional(Field(&MockSummaryData2::Text, "twenty")));
  EXPECT_THAT(getAsEntitySummary<MockSummaryData3>(Summary, Name4, ID2),
              Optional(Field(&MockSummaryData3::Flag, true)));
}

} // namespace
