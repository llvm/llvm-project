//===- SharedLexicalRepresentationFormatTest.cpp --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "JSONFormatTest.h"

#include "clang/ScalableStaticAnalysis/Analyses/SharedLexicalRepresentation/SharedLexicalRepresentation.h"
#include "clang/ScalableStaticAnalysis/Core/EntityLinker/LUSummary.h"
#include "clang/ScalableStaticAnalysis/Core/Model/BuildNamespace.h"
#include "clang/ScalableStaticAnalysis/Core/Model/EntityId.h"
#include "clang/ScalableStaticAnalysis/Core/Model/EntityLinkage.h"
#include "clang/ScalableStaticAnalysis/Core/Model/EntityName.h"
#include "clang/ScalableStaticAnalysis/Core/Serialization/JSONFormat.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"

#include <memory>
#include <utility>
#include <vector>

namespace clang::ssaf {

extern EntitySourceLocationsSummary
buildEntitySourceLocationsSummary(std::vector<SourceLocationRecord> Locs);

namespace {

TEST(SourceLocationRecordTest, DefaultConstructed) {
  SourceLocationRecord R;
  EXPECT_TRUE(R.FilePath.empty());
  EXPECT_EQ(R.Line, 0u);
  EXPECT_EQ(R.Column, 0u);
}

TEST(SourceLocationRecordTest, EqualityIsTupleWise) {
  SourceLocationRecord A{"/a/b.h", 5, 13};
  SourceLocationRecord B{"/a/b.h", 5, 13};
  SourceLocationRecord C{"/a/b.h", 5, 14};
  SourceLocationRecord D{"/a/b.h", 6, 13};
  SourceLocationRecord E{"/a/c.h", 5, 13};
  EXPECT_EQ(A, B);
  EXPECT_FALSE(A == C);
  EXPECT_FALSE(A == D);
  EXPECT_FALSE(A == E);
}

TEST(SourceLocationRecordTest, OrderingIsLexicographic) {
  SourceLocationRecord A{"/a/b.h", 5, 13};
  SourceLocationRecord B{"/a/b.h", 5, 14};
  SourceLocationRecord C{"/a/b.h", 6, 0};
  SourceLocationRecord D{"/a/c.h", 1, 1};
  EXPECT_TRUE(A < B);
  EXPECT_TRUE(B < C);
  EXPECT_TRUE(C < D);
  EXPECT_FALSE(B < A);
  EXPECT_FALSE(A < A);
}

TEST(EntitySourceLocationsSummaryTest, SummaryName) {
  EXPECT_EQ(EntitySourceLocationsSummary::summaryName(),
            SummaryName("EntitySourceLocations"));
  EntitySourceLocationsSummary S = buildEntitySourceLocationsSummary({});
  EXPECT_EQ(S.getSummaryName(), EntitySourceLocationsSummary::summaryName());
}

TEST(EntitySourceLocationsSummaryTest, EmptyByDefault) {
  EntitySourceLocationsSummary S = buildEntitySourceLocationsSummary({});
  EXPECT_TRUE(S.empty());
}

class SharedLexicalRepresentationFormatTest : public JSONFormatTest {
protected:
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

  void insertSummary(LUSummary &LU, EntityId Id,
                     std::vector<SourceLocationRecord> Locs) {
    getData(LU)[EntitySourceLocationsSummary::summaryName()][Id] =
        std::make_unique<EntitySourceLocationsSummary>(
            buildEntitySourceLocationsSummary(std::move(Locs)));
  }

  llvm::Expected<LUSummary> roundTrip(const LUSummary &LU) {
    PathString Path = makePath("slr-summary.json");
    if (auto Err = JSONFormat().writeLUSummary(LU, Path))
      return std::move(Err);
    return JSONFormat().readLUSummary(Path);
  }

  static const EntitySourceLocationsSummary *findSummary(const LUSummary &LU,
                                                         EntityId Id) {
    const auto &Data = getData(LU);
    auto It = Data.find(EntitySourceLocationsSummary::summaryName());
    if (It == Data.end())
      return nullptr;
    auto It2 = It->second.find(Id);
    if (It2 == It->second.end())
      return nullptr;
    return static_cast<const EntitySourceLocationsSummary *>(It2->second.get());
  }
};

TEST_F(SharedLexicalRepresentationFormatTest, EmptySummaryRoundTrips) {
  auto LU = makeLUSummary();
  EntityId E = addEntity(*LU, "c:@p");
  insertSummary(*LU, E, {});

  auto Round = roundTrip(*LU);
  ASSERT_THAT_EXPECTED(Round, llvm::Succeeded());

  const auto *Out = findSummary(*Round, E);
  ASSERT_NE(Out, nullptr);
  EXPECT_TRUE(Out->empty());
}

TEST_F(SharedLexicalRepresentationFormatTest, SingleRecordRoundTrips) {
  auto LU = makeLUSummary();
  EntityId E = addEntity(*LU, "c:@p");
  std::vector<SourceLocationRecord> Locs{{"/abs/path/foo.h", 5, 13}};
  EntitySourceLocationsSummary Expected =
      buildEntitySourceLocationsSummary(Locs);
  insertSummary(*LU, E, std::move(Locs));

  auto Round = roundTrip(*LU);
  ASSERT_THAT_EXPECTED(Round, llvm::Succeeded());

  const auto *Out = findSummary(*Round, E);
  ASSERT_NE(Out, nullptr);
  EXPECT_EQ(*Out, Expected);
}

TEST_F(SharedLexicalRepresentationFormatTest, MultiRecordRoundTrips) {
  auto LU = makeLUSummary();
  EntityId E = addEntity(*LU, "c:@p");
  std::vector<SourceLocationRecord> Locs{{"/abs/header.h", 5, 13},
                                         {"/abs/source.cpp", 1, 1},
                                         {"/abs/header.h", 5, 13}};
  EntitySourceLocationsSummary Expected =
      buildEntitySourceLocationsSummary(Locs);
  insertSummary(*LU, E, std::move(Locs));

  auto Round = roundTrip(*LU);
  ASSERT_THAT_EXPECTED(Round, llvm::Succeeded());

  const auto *Out = findSummary(*Round, E);
  ASSERT_NE(Out, nullptr);
  EXPECT_EQ(*Out, Expected);
}

TEST_F(SharedLexicalRepresentationFormatTest, MultiEntitySharedLocRoundTrips) {
  auto LU = makeLUSummary();
  EntityId E1 = addEntity(*LU, "c:@p");
  EntityId E2 = addEntity(*LU, "c:@q");
  std::vector<SourceLocationRecord> Locs1{{"/abs/header.h", 5, 13}};
  std::vector<SourceLocationRecord> Locs2{{"/abs/header.h", 5, 13},
                                          {"/abs/source.cpp", 10, 5}};
  EntitySourceLocationsSummary Expected1 =
      buildEntitySourceLocationsSummary(Locs1);
  EntitySourceLocationsSummary Expected2 =
      buildEntitySourceLocationsSummary(Locs2);
  insertSummary(*LU, E1, std::move(Locs1));
  insertSummary(*LU, E2, std::move(Locs2));

  auto Round = roundTrip(*LU);
  ASSERT_THAT_EXPECTED(Round, llvm::Succeeded());

  const auto *Out1 = findSummary(*Round, E1);
  const auto *Out2 = findSummary(*Round, E2);
  ASSERT_NE(Out1, nullptr);
  ASSERT_NE(Out2, nullptr);
  EXPECT_EQ(*Out1, Expected1);
  EXPECT_EQ(*Out2, Expected2);
}

TEST_F(SharedLexicalRepresentationFormatTest, WireFormatKeys) {
  auto LU = makeLUSummary();
  EntityId E = addEntity(*LU, "c:@p");
  insertSummary(*LU, E, {{"/abs/path/foo.h", 5, 13}});

  PathString Path = makePath("slr-summary.json");
  ASSERT_THAT_ERROR(JSONFormat().writeLUSummary(*LU, Path), llvm::Succeeded());

  auto JSON = readJSONFromFile("slr-summary.json");
  ASSERT_THAT_EXPECTED(JSON, llvm::Succeeded());

  const llvm::json::Object *Root = JSON->getAsObject();
  ASSERT_NE(Root, nullptr);
  const llvm::json::Array *Data = Root->getArray("data");
  ASSERT_NE(Data, nullptr);

  const llvm::json::Array *Entries = nullptr;
  for (const llvm::json::Value &Entry : *Data) {
    const llvm::json::Object *EntryObj = Entry.getAsObject();
    ASSERT_NE(EntryObj, nullptr);
    auto Name = EntryObj->getString("summary_name");
    ASSERT_TRUE(Name.has_value());
    if (*Name == "EntitySourceLocations") {
      Entries = EntryObj->getArray("summary_data");
      break;
    }
  }
  ASSERT_NE(Entries, nullptr);
  ASSERT_EQ(Entries->size(), 1u);

  const llvm::json::Object *EntryObj = (*Entries)[0].getAsObject();
  ASSERT_NE(EntryObj, nullptr);
  const llvm::json::Object *SummaryObj = EntryObj->getObject("entity_summary");
  ASSERT_NE(SummaryObj, nullptr);
  const llvm::json::Array *Locs = SummaryObj->getArray("decl_locations");
  ASSERT_NE(Locs, nullptr);
  ASSERT_EQ(Locs->size(), 1u);

  const llvm::json::Object *Loc = (*Locs)[0].getAsObject();
  ASSERT_NE(Loc, nullptr);
  EXPECT_EQ(Loc->getString("file_path"), "/abs/path/foo.h");
  EXPECT_EQ(Loc->getInteger("line"), 5);
  EXPECT_EQ(Loc->getInteger("column"), 13);
}

TEST_F(SharedLexicalRepresentationFormatTest, BoundaryValuesRoundTrip) {
  auto LU = makeLUSummary();
  EntityId E = addEntity(*LU, "c:@boundary");
  std::vector<SourceLocationRecord> Locs{
      {"", 0, 0}, {"/path/with spaces/foo.h", 1000000, 65535}};
  EntitySourceLocationsSummary Expected =
      buildEntitySourceLocationsSummary(Locs);
  insertSummary(*LU, E, std::move(Locs));

  auto Round = roundTrip(*LU);
  ASSERT_THAT_EXPECTED(Round, llvm::Succeeded());

  const auto *Out = findSummary(*Round, E);
  ASSERT_NE(Out, nullptr);
  EXPECT_EQ(*Out, Expected);
}

} // namespace

} // namespace clang::ssaf
