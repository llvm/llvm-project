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
#include <memory>
#include <type_traits>

using namespace clang;
using namespace ssaf;

using llvm::SmallVector;
using testing::Field;
using testing::Optional;
using testing::UnorderedElementsAre;

[[nodiscard]]
static EntityId addTestEntity(TUSummaryBuilder &Builder, llvm::StringRef USR) {
  return Builder.addEntity(EntityName(USR, /*Suffix=*/"", /*Namespace=*/{}));
}

struct FactResult {
  EntitySummary *Fact;
  bool Inserted;
};

template <class ConcreteEntitySummary>
[[nodiscard]]
static auto addFactTo(TUSummaryBuilder &Builder, EntityId ID,
                      ConcreteEntitySummary Fact) {
  static_assert(std::is_base_of_v<EntitySummary, ConcreteEntitySummary>);
  auto NewFact = std::make_unique<ConcreteEntitySummary>(std::move(Fact));
  SummaryName Name = NewFact->getSummaryName();
  auto [Place, Inserted] = Builder.addFact(ID, std::move(NewFact));
  return std::pair{Name, FactResult{Place, Inserted}};
}

namespace {

// Mock EntitySummary classes for testing
struct MockSummaryData1 final : public EntitySummary {
  explicit MockSummaryData1(int Value) : Value(Value) {}
  SummaryName getSummaryName() const override {
    return SummaryName("MockSummary1");
  }
  int Value;
};

struct MockSummaryData2 final : public EntitySummary {
  explicit MockSummaryData2(std::string Text) : Text(std::move(Text)) {}
  SummaryName getSummaryName() const override {
    return SummaryName("MockSummary2");
  }
  std::string Text;
};

struct MockSummaryData3 final : public EntitySummary {
  explicit MockSummaryData3(bool Flag) : Flag(Flag) {}
  SummaryName getSummaryName() const override {
    return SummaryName("MockSummary3");
  }
  bool Flag;
};

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
  TUSummary Summary =
      BuildNamespace(BuildNamespaceKind::CompilationUnit, "Mock.cpp");
  TUSummaryBuilder Builder = TUSummaryBuilder(this->Summary);

  [[nodiscard]] static SmallVector<SummaryName>
  summaryNames(const TUSummary &Summary) {
    return llvm::to_vector(llvm::make_first_range(getData(Summary)));
  }

  [[nodiscard]] static SmallVector<EntityId>
  entitiesOfSummary(const TUSummary &Summary, const SummaryName &Name) {
    const auto &MappingIt = getData(Summary).find(Name);
    if (MappingIt == getData(Summary).end())
      return {};
    return llvm::to_vector(llvm::make_first_range(MappingIt->second));
  }

  template <class ConcreteSummaryData>
  [[nodiscard]] static std::optional<ConcreteSummaryData>
  getAsEntitySummary(const TUSummary &Summary, const SummaryName &Name,
                     EntityId E) {
    static_assert(std::is_base_of_v<EntitySummary, ConcreteSummaryData>);
    const auto &MappingIt = getData(Summary).find(Name);
    if (MappingIt == getData(Summary).end())
      return std::nullopt;
    auto SummaryIt = MappingIt->second.find(E);
    if (SummaryIt == MappingIt->second.end())
      return std::nullopt;
    assert(Name == SummaryIt->second->getSummaryName());
    return static_cast<const ConcreteSummaryData &>(*SummaryIt->second);
  }
};

TEST_F(TUSummaryBuilderTest, AddEntity) {
  EntityName EN1("c:@F@foo", "", /*Namespace=*/{});
  EntityName EN2("c:@F@bar", "", /*Namespace=*/{});

  EntityId ID = Builder.addEntity(EN1);
  EntityId IDAlias = Builder.addEntity(EN1);
  EXPECT_EQ(ID, IDAlias); // Idenpotency

  EntityId ID2 = Builder.addEntity(EN2);
  EXPECT_NE(ID, ID2);
  EXPECT_NE(IDAlias, ID2);

  const EntityIdTable &IdTable = getIdTable(Summary);
  EXPECT_EQ(IdTable.count(), 2U);
  EXPECT_TRUE(IdTable.contains(EN1));
  EXPECT_TRUE(IdTable.contains(EN2));
}

TEST_F(TUSummaryBuilderTest, TUSummaryBuilderAddSingleFact) {
  EntityId ID = addTestEntity(Builder, "c:@F@foo");
  auto [Name, Res] = addFactTo(Builder, ID, MockSummaryData1(10));
  ASSERT_TRUE(Res.Inserted);
  ASSERT_TRUE(Res.Fact);

  // Should have a summary type with an entity.
  EXPECT_THAT(summaryNames(Summary), UnorderedElementsAre(Name));
  EXPECT_THAT(entitiesOfSummary(Summary, Name), UnorderedElementsAre(ID));

  EXPECT_THAT(getAsEntitySummary<MockSummaryData1>(Summary, Name, ID),
              Optional(Field(&MockSummaryData1::Value, 10)));
}

TEST_F(TUSummaryBuilderTest, AddMultipleFactsToSameEntity) {
  EntityId ID = addTestEntity(Builder, "c:@F@foo");

  // Add different summary types to the same entity.
  auto [Name1, Res1] = addFactTo(Builder, ID, MockSummaryData1(42));
  auto [Name2, Res2] = addFactTo(Builder, ID, MockSummaryData2("test data"));
  auto [Name3, Res3] = addFactTo(Builder, ID, MockSummaryData3(true));
  ASSERT_TRUE(Res1.Inserted);
  ASSERT_TRUE(Res2.Inserted);
  ASSERT_TRUE(Res3.Inserted);
  ASSERT_TRUE(Res1.Fact);
  ASSERT_TRUE(Res2.Fact);
  ASSERT_TRUE(Res3.Fact);

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
  EntityId ID1 = addTestEntity(Builder, "c:@F@foo");
  EntityId ID2 = addTestEntity(Builder, "c:@F@bar");
  EntityId ID3 = addTestEntity(Builder, "c:@F@baz");

  // Add the same summary type to different entities.
  auto [Name1, Res1] = addFactTo(Builder, ID1, MockSummaryData1(1));
  auto [Name2, Res2] = addFactTo(Builder, ID2, MockSummaryData1(2));
  auto [Name3, Res3] = addFactTo(Builder, ID3, MockSummaryData1(3));
  ASSERT_TRUE(Res1.Inserted);
  ASSERT_TRUE(Res2.Inserted);
  ASSERT_TRUE(Res3.Inserted);
  ASSERT_TRUE(Res1.Fact);
  ASSERT_TRUE(Res2.Fact);
  ASSERT_TRUE(Res3.Fact);

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

TEST_F(TUSummaryBuilderTest, AddConflictingFactToSameEntity) {
  EntityId ID = addTestEntity(Builder, "c:@F@foo");

  auto [Name, Res] = addFactTo(Builder, ID, MockSummaryData1(10));
  ASSERT_TRUE(Res.Inserted);
  ASSERT_TRUE(Res.Fact);

  // Check the initial value.
  EXPECT_THAT(summaryNames(Summary), UnorderedElementsAre(Name));
  EXPECT_THAT(entitiesOfSummary(Summary, Name), UnorderedElementsAre(ID));
  EXPECT_THAT(getAsEntitySummary<MockSummaryData1>(Summary, Name, ID),
              Optional(Field(&MockSummaryData1::Value, 10)));

  // This is a different fact of the same kind.
  std::unique_ptr<EntitySummary> NewFact =
      std::make_unique<MockSummaryData1>(20);
  ASSERT_EQ(NewFact->getSummaryName(), Name);

  // Let's add this different fact.
  // This should keep the map intact and give us the existing entity summary.
  auto [Slot, Inserted] = Builder.addFact(ID, std::move(NewFact));
  ASSERT_FALSE(Inserted);
  ASSERT_TRUE(Slot);

  // Check that the fact object is not consumed and remained the same.
  ASSERT_TRUE(NewFact);
  ASSERT_EQ(static_cast<MockSummaryData1 *>(NewFact.get())->Value, 20);

  // Check that the Slot refers to the existing entity summary.
  ASSERT_EQ(Slot->getSummaryName(), Name);
  ASSERT_EQ(static_cast<MockSummaryData1 *>(Slot)->Value, 10);

  // Check that the values remained the same.
  EXPECT_THAT(summaryNames(Summary), UnorderedElementsAre(Name));
  EXPECT_THAT(entitiesOfSummary(Summary, Name), UnorderedElementsAre(ID));
  EXPECT_THAT(getAsEntitySummary<MockSummaryData1>(Summary, Name, ID),
              Optional(Field(&MockSummaryData1::Value, 10)));

  // We can update the existing summary.
  static_cast<MockSummaryData1 *>(Res.Fact)->Value = 30;

  // Check that the values remained the same except what we updated.
  EXPECT_THAT(summaryNames(Summary), UnorderedElementsAre(Name));
  EXPECT_THAT(entitiesOfSummary(Summary, Name), UnorderedElementsAre(ID));
  EXPECT_THAT(getAsEntitySummary<MockSummaryData1>(Summary, Name, ID),
              Optional(Field(&MockSummaryData1::Value, 30)));
}

} // namespace
