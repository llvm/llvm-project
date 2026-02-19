//===- unittests/Analysis/Scalable/EntityLinkerTest.cpp ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception.
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/Scalable/EntityLinker/EntityLinker.h"
#include "TestFixture.h"
#include "clang/Analysis/Scalable/EntityLinker/EntitySummaryEncoding.h"
#include "clang/Analysis/Scalable/EntityLinker/LUSummaryEncoding.h"
#include "clang/Analysis/Scalable/EntityLinker/TUSummaryEncoding.h"
#include "clang/Analysis/Scalable/Model/BuildNamespace.h"
#include "clang/Analysis/Scalable/Model/EntityId.h"
#include "clang/Analysis/Scalable/Model/EntityIdTable.h"
#include "clang/Analysis/Scalable/Model/EntityLinkage.h"
#include "clang/Analysis/Scalable/Model/EntityName.h"
#include "clang/Analysis/Scalable/Model/SummaryName.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"
#include <memory>

using namespace clang::ssaf;
using namespace llvm;

namespace {

class MockEntitySummaryEncoding : public EntitySummaryEncoding {
public:
  MockEntitySummaryEncoding() : Id(++Index) {}

  size_t getId() const { return Id; }

  void
  patch(const std::map<EntityId, EntityId> &EntityResolutionTable) override {
    PatchedIds = EntityResolutionTable;
  }

  const std::map<EntityId, EntityId> &getPatchedIds() const {
    return PatchedIds;
  }

  static size_t Index;

private:
  size_t Id;
  std::map<EntityId, EntityId> PatchedIds;
};

size_t MockEntitySummaryEncoding::Index = 0;

class EntityLinkerTest : public TestFixture {
protected:
  std::unique_ptr<TUSummaryEncoding>
  createTUSummaryEncoding(BuildNamespaceKind Kind, llvm::StringRef Name) {
    return std::make_unique<TUSummaryEncoding>(BuildNamespace(Kind, Name));
  }

  size_t addSummaryData(TUSummaryEncoding &TU, EntityId EId,
                        llvm::StringRef SummaryNameStr) {
    SummaryName SN(SummaryNameStr.str());
    auto Summary = std::make_unique<MockEntitySummaryEncoding>();
    const size_t ESId = Summary->getId();
    getData(TU)[SN][EId] = std::move(Summary);
    return ESId;
  }

  EntityId addEntity(TUSummaryEncoding &TU, llvm::StringRef USR,
                     EntityLinkage::LinkageType Linkage) {
    EntityName Name(USR, "", NestedBuildNamespace(getTUNamespace(TU)));
    EntityId Id = getIdTable(TU).getId(Name);
    getLinkageTable(TU).insert({Id, EntityLinkage(Linkage)});
    return Id;
  }
};

// ============================================================================
// Entity ID Table Matchers
// ============================================================================

MATCHER_P(ContainsEntity, entityName,
          std::string(negation ? "does not contain" : "contains") +
              " entity with name '" + ::testing::PrintToString(entityName) +
              "'") {
  return arg.contains(entityName);
}

MATCHER_P(IdTableHasSize, expectedCount,
          std::string("has ") + ::testing::PrintToString(expectedCount) +
              " entities") {
  if (arg.count() != expectedCount) {
    *result_listener << "has " << arg.count() << " entities";
    return false;
  }
  return true;
}

// ============================================================================
// Linkage Table Matchers
// ============================================================================

MATCHER_P2(EntityHasLinkage, entityId, expectedLinkage,
           std::string("entity has ") +
               ::testing::PrintToString(expectedLinkage) + " linkage") {
  auto it = arg.find(entityId);
  if (it == arg.end()) {
    *result_listener << "entity " << ::testing::PrintToString(entityId)
                     << " not found in linkage table";
    return false;
  }

  auto actualLinkage = it->second.getLinkage();
  if (actualLinkage != expectedLinkage) {
    *result_listener << "entity " << ::testing::PrintToString(entityId)
                     << " has linkage "
                     << ::testing::PrintToString(actualLinkage);
    return false;
  }

  return true;
}

MATCHER_P(LinkageTableHasSize, expectedSize,
          std::string("linkage table has size ") +
              ::testing::PrintToString(expectedSize)) {
  if (arg.size() != expectedSize) {
    *result_listener << "has size " << arg.size();
    return false;
  }
  return true;
}

// ============================================================================
// Summary Data Matchers
// ============================================================================

MATCHER_P3(HasSummaryData, entityId, expectedMockId, expectedResolutionMapping,
           std::string("has summary data for entity with expected mock ID ") +
               ::testing::PrintToString(expectedMockId)) {

  auto it = arg.find(entityId);
  if (it == arg.end()) {
    *result_listener << "entity " << ::testing::PrintToString(entityId)
                     << " not found in summary data";
    return false;
  }

  auto *mock = static_cast<const MockEntitySummaryEncoding *>(it->second.get());

  if (mock->getId() != expectedMockId) {
    *result_listener << "entity " << ::testing::PrintToString(entityId)
                     << " has mock ID " << mock->getId() << " (expected "
                     << expectedMockId << ")";
    return false;
  }

  if (mock->getPatchedIds() != expectedResolutionMapping) {
    *result_listener << "entity " << ::testing::PrintToString(entityId)
                     << " has different resolution mapping";
    return false;
  }

  return true;
}

MATCHER_P(DoesNotContainSummaryFor, entityId,
          std::string("does not contain summary data for entity")) {
  if (arg.find(entityId) != arg.end()) {
    *result_listener << "unexpectedly contains entity "
                     << ::testing::PrintToString(entityId);
    return false;
  }
  return true;
}

MATCHER_P(SummaryDataHasSize, expectedSize,
          std::string("summary data has size ") +
              ::testing::PrintToString(expectedSize)) {
  if (arg.size() != expectedSize) {
    *result_listener << "has size " << arg.size();
    return false;
  }
  return true;
}

TEST_F(EntityLinkerTest, CreatesEmptyLinker) {
  NestedBuildNamespace LUNamespace(
      {BuildNamespace(BuildNamespaceKind::LinkUnit, "LU")});

  EntityLinker Linker(LUNamespace);

  const auto &Output = Linker.getOutput();
  EXPECT_EQ(getIdTable(Output).count(), 0u);
  EXPECT_EQ(getLinkageTable(Output).size(), 0u);
  EXPECT_EQ(getData(Output).size(), 0u);
}

TEST_F(EntityLinkerTest, LinksEmptyTranslationUnit) {
  NestedBuildNamespace LUNamespace(
      {BuildNamespace(BuildNamespaceKind::LinkUnit, "LU")});

  EntityLinker Linker(LUNamespace);

  auto TUEmpty =
      createTUSummaryEncoding(BuildNamespaceKind::CompilationUnit, "TUEmpty");

  EXPECT_THAT_ERROR(Linker.link(std::move(TUEmpty)), llvm::Succeeded());

  const auto &Output = Linker.getOutput();
  EXPECT_EQ(getIdTable(Output).count(), 0u);
  EXPECT_EQ(getLinkageTable(Output).size(), 0u);
  EXPECT_EQ(getData(Output).size(), 0u);
}

TEST_F(EntityLinkerTest, LinksOneTranslationUnit) {
  NestedBuildNamespace LUNamespace(
      {BuildNamespace(BuildNamespaceKind::LinkUnit, "LU")});

  EntityLinker Linker(LUNamespace);

  auto TU = createTUSummaryEncoding(BuildNamespaceKind::CompilationUnit, "TU");

  const auto TU_A_Id = addEntity(*TU, "A", EntityLinkage::LinkageType::None);
  const auto TU_A_S1_Data = addSummaryData(*TU, TU_A_Id, "S1");
  const auto TU_A_S2_Data = addSummaryData(*TU, TU_A_Id, "S2");

  const auto TU_B_Id =
      addEntity(*TU, "B", EntityLinkage::LinkageType::Internal);
  const auto TU_B_S1_Data = addSummaryData(*TU, TU_B_Id, "S1");
  const auto TU_B_S2_Data = addSummaryData(*TU, TU_B_Id, "S2");

  const auto TU_C_Id =
      addEntity(*TU, "C", EntityLinkage::LinkageType::External);
  const auto TU_C_S1_Data = addSummaryData(*TU, TU_C_Id, "S1");

  const auto TU_D_Id =
      addEntity(*TU, "D", EntityLinkage::LinkageType::External);
  const auto TU_D_S2_Data = addSummaryData(*TU, TU_D_Id, "S2");

  const BuildNamespace TUNamespace = getTUNamespace(*TU);

  ASSERT_THAT_ERROR(Linker.link(std::move(TU)), llvm::Succeeded());

  const auto &Output = Linker.getOutput();
  const auto &IdTable = getIdTable(Output);
  const auto &Entities = getEntities(IdTable);
  const auto &LinkageTable = getLinkageTable(Output);
  const auto &Data = getData(Output);

  NestedBuildNamespace LocalNamespace =
      NestedBuildNamespace(TUNamespace).makeQualified(LUNamespace);

  EntityName LU_A_Name("A", "", LocalNamespace);
  EntityName LU_B_Name("B", "", LocalNamespace);
  EntityName LU_C_Name("C", "", LUNamespace);
  EntityName LU_D_Name("D", "", LUNamespace);

  // EntityIdTable Tests.
  {
    ASSERT_THAT(IdTable, IdTableHasSize(4u));
    ASSERT_THAT(IdTable, ContainsEntity(LU_A_Name));
    ASSERT_THAT(IdTable, ContainsEntity(LU_B_Name));
    ASSERT_THAT(IdTable, ContainsEntity(LU_C_Name));
    ASSERT_THAT(IdTable, ContainsEntity(LU_D_Name));
  }

  // This is safe since we confirmed that these entities are present in the
  // block above.
  const auto LU_A_Id = Entities.at(LU_A_Name);
  const auto LU_B_Id = Entities.at(LU_B_Name);
  const auto LU_C_Id = Entities.at(LU_C_Name);
  const auto LU_D_Id = Entities.at(LU_D_Name);

  // LinkageTable Tests.
  {
    ASSERT_THAT(LinkageTable, LinkageTableHasSize(4u));
    ASSERT_THAT(LinkageTable,
                EntityHasLinkage(LU_A_Id, EntityLinkage::LinkageType::None));
    ASSERT_THAT(
        LinkageTable,
        EntityHasLinkage(LU_B_Id, EntityLinkage::LinkageType::Internal));
    ASSERT_THAT(
        LinkageTable,
        EntityHasLinkage(LU_C_Id, EntityLinkage::LinkageType::External));
    ASSERT_THAT(
        LinkageTable,
        EntityHasLinkage(LU_D_Id, EntityLinkage::LinkageType::External));
  }

  std::map<EntityId, EntityId> Resolution = {{TU_A_Id, LU_A_Id},
                                             {TU_B_Id, LU_B_Id},
                                             {TU_C_Id, LU_C_Id},
                                             {TU_D_Id, LU_D_Id}};

  // Data Tests.
  {
    ASSERT_EQ(Data.size(), 2u);

    // S1 Data Tests.
    {
      SummaryName S1("S1");
      ASSERT_NE(Data.find(S1), Data.end());
      const auto &S1Data = Data.at(S1);

      ASSERT_THAT(S1Data, SummaryDataHasSize(3u));
      ASSERT_THAT(S1Data, HasSummaryData(LU_A_Id, TU_A_S1_Data, Resolution));
      ASSERT_THAT(S1Data, HasSummaryData(LU_B_Id, TU_B_S1_Data, Resolution));
      ASSERT_THAT(S1Data, HasSummaryData(TU_C_Id, TU_C_S1_Data, Resolution));
    }

    // S2 Data Tests.
    {
      SummaryName S2("S2");
      ASSERT_NE(Data.find(S2), Data.end());
      const auto &S2Data = Data.at(S2);

      ASSERT_THAT(S2Data, SummaryDataHasSize(3u));
      ASSERT_THAT(S2Data, HasSummaryData(LU_A_Id, TU_A_S2_Data, Resolution));
      ASSERT_THAT(S2Data, HasSummaryData(LU_B_Id, TU_B_S2_Data, Resolution));
      ASSERT_THAT(S2Data, HasSummaryData(TU_D_Id, TU_D_S2_Data, Resolution));
    }
  }
}

TEST_F(EntityLinkerTest, LinksTwoTranslationUnits) {
  NestedBuildNamespace LUNamespace(
      {BuildNamespace(BuildNamespaceKind::LinkUnit, "LU")});

  EntityLinker Linker(LUNamespace);

  auto TU1 =
      createTUSummaryEncoding(BuildNamespaceKind::CompilationUnit, "TU1");

  // None linkage entities in TU1
  const auto TU1_X_Id = addEntity(*TU1, "X", EntityLinkage::LinkageType::None);
  const auto TU1_X_S1_Data = addSummaryData(*TU1, TU1_X_Id, "S1");

  const auto TU1_Y_Id = addEntity(*TU1, "Y", EntityLinkage::LinkageType::None);
  const auto TU1_Y_S2_Data = addSummaryData(*TU1, TU1_Y_Id, "S2");

  const auto TU1_Z_Id = addEntity(*TU1, "Z", EntityLinkage::LinkageType::None);
  const auto TU1_Z_S1_Data = addSummaryData(*TU1, TU1_Z_Id, "S1");
  const auto TU1_Z_S2_Data = addSummaryData(*TU1, TU1_Z_Id, "S2");

  // Internal linkage entities in TU1
  const auto TU1_A_Id =
      addEntity(*TU1, "A", EntityLinkage::LinkageType::Internal);
  const auto TU1_A_S1_Data = addSummaryData(*TU1, TU1_A_Id, "S1");

  const auto TU1_B_Id =
      addEntity(*TU1, "B", EntityLinkage::LinkageType::Internal);
  const auto TU1_B_S2_Data = addSummaryData(*TU1, TU1_B_Id, "S2");

  const auto TU1_C_Id =
      addEntity(*TU1, "C", EntityLinkage::LinkageType::Internal);
  const auto TU1_C_S1_Data = addSummaryData(*TU1, TU1_C_Id, "S1");
  const auto TU1_C_S2_Data = addSummaryData(*TU1, TU1_C_Id, "S2");

  // External linkage entities in TU1
  const auto TU1_P_Id =
      addEntity(*TU1, "P", EntityLinkage::LinkageType::External);
  const auto TU1_P_S1_Data = addSummaryData(*TU1, TU1_P_Id, "S1");

  const auto TU1_Q_Id =
      addEntity(*TU1, "Q", EntityLinkage::LinkageType::External);
  const auto TU1_Q_S2_Data = addSummaryData(*TU1, TU1_Q_Id, "S2");

  const auto TU1_R_Id =
      addEntity(*TU1, "R", EntityLinkage::LinkageType::External);
  const auto TU1_R_S1_Data = addSummaryData(*TU1, TU1_R_Id, "S1");
  const auto TU1_R_S2_Data = addSummaryData(*TU1, TU1_R_Id, "S2");

  const BuildNamespace TU1Namespace = getTUNamespace(*TU1);

  ASSERT_THAT_ERROR(Linker.link(std::move(TU1)), llvm::Succeeded());

  auto TU2 =
      createTUSummaryEncoding(BuildNamespaceKind::CompilationUnit, "TU2");

  // None linkage entities in TU2 - includes duplicates and uniques
  const auto TU2_X_Id = addEntity(*TU2, "X", EntityLinkage::LinkageType::None);
  const auto TU2_X_S2_Data = addSummaryData(*TU2, TU2_X_Id, "S2");

  const auto TU2_Y_Id = addEntity(*TU2, "Y", EntityLinkage::LinkageType::None);
  const auto TU2_Y_S1_Data = addSummaryData(*TU2, TU2_Y_Id, "S1");

  const auto TU2_W_Id = addEntity(*TU2, "W", EntityLinkage::LinkageType::None);
  const auto TU2_W_S1_Data = addSummaryData(*TU2, TU2_W_Id, "S1");
  const auto TU2_W_S2_Data = addSummaryData(*TU2, TU2_W_Id, "S2");

  // Internal linkage entities in TU2 - includes duplicates and unique
  const auto TU2_A_Id =
      addEntity(*TU2, "A", EntityLinkage::LinkageType::Internal);
  const auto TU2_A_S2_Data = addSummaryData(*TU2, TU2_A_Id, "S2");

  const auto TU2_B_Id =
      addEntity(*TU2, "B", EntityLinkage::LinkageType::Internal);
  const auto TU2_B_S1_Data = addSummaryData(*TU2, TU2_B_Id, "S1");

  const auto TU2_D_Id =
      addEntity(*TU2, "D", EntityLinkage::LinkageType::Internal);
  const auto TU2_D_S1_Data = addSummaryData(*TU2, TU2_D_Id, "S1");
  const auto TU2_D_S2_Data = addSummaryData(*TU2, TU2_D_Id, "S2");

  // External linkage entities in TU2 - includes duplicates (will be dropped)
  // and uniques
  const auto TU2_P_Id =
      addEntity(*TU2, "P", EntityLinkage::LinkageType::External);
  const auto TU2_P_S2_Data = addSummaryData(*TU2, TU2_P_Id, "S2");

  const auto TU2_Q_Id =
      addEntity(*TU2, "Q", EntityLinkage::LinkageType::External);
  const auto TU2_Q_S1_Data = addSummaryData(*TU2, TU2_Q_Id, "S1");

  const auto TU2_S_Id =
      addEntity(*TU2, "S", EntityLinkage::LinkageType::External);
  const auto TU2_S_S1_Data = addSummaryData(*TU2, TU2_S_Id, "S1");
  const auto TU2_S_S2_Data = addSummaryData(*TU2, TU2_S_Id, "S2");

  const BuildNamespace TU2Namespace = getTUNamespace(*TU2);

  ASSERT_THAT_ERROR(Linker.link(std::move(TU2)), llvm::Succeeded());

  const auto &Output = Linker.getOutput();
  const auto &IdTable = getIdTable(Output);
  const auto &Entities = getEntities(IdTable);
  const auto &LinkageTable = getLinkageTable(Output);
  const auto &Data = getData(Output);

  NestedBuildNamespace TU1LocalNamespace =
      NestedBuildNamespace(TU1Namespace).makeQualified(LUNamespace);

  NestedBuildNamespace TU2LocalNamespace =
      NestedBuildNamespace(TU2Namespace).makeQualified(LUNamespace);

  // None linkage entities use local namespace (TU scoped)
  EntityName LU_TU1_X_Name("X", "", TU1LocalNamespace);
  EntityName LU_TU1_Y_Name("Y", "", TU1LocalNamespace);
  EntityName LU_TU1_Z_Name("Z", "", TU1LocalNamespace);
  EntityName LU_TU2_X_Name("X", "", TU2LocalNamespace);
  EntityName LU_TU2_Y_Name("Y", "", TU2LocalNamespace);
  EntityName LU_TU2_W_Name("W", "", TU2LocalNamespace);

  // Internal linkage entities use local namespace (TU scoped)
  EntityName LU_TU1_A_Name("A", "", TU1LocalNamespace);
  EntityName LU_TU1_B_Name("B", "", TU1LocalNamespace);
  EntityName LU_TU1_C_Name("C", "", TU1LocalNamespace);
  EntityName LU_TU2_A_Name("A", "", TU2LocalNamespace);
  EntityName LU_TU2_B_Name("B", "", TU2LocalNamespace);
  EntityName LU_TU2_D_Name("D", "", TU2LocalNamespace);

  // External linkage entities use LU namespace (shared across TUs)
  EntityName LU_P_Name("P", "", LUNamespace);
  EntityName LU_Q_Name("Q", "", LUNamespace);
  EntityName LU_R_Name("R", "", LUNamespace);
  EntityName LU_S_Name("S", "", LUNamespace);

  // EntityIdTable Tests.
  {
    // Should have 6 None + 6 Internal + 4 External = 16 entities total
    ASSERT_THAT(IdTable, IdTableHasSize(16u));

    ASSERT_THAT(IdTable, ContainsEntity(LU_TU1_X_Name));
    ASSERT_THAT(IdTable, ContainsEntity(LU_TU1_Y_Name));
    ASSERT_THAT(IdTable, ContainsEntity(LU_TU1_Z_Name));
    ASSERT_THAT(IdTable, ContainsEntity(LU_TU2_X_Name));
    ASSERT_THAT(IdTable, ContainsEntity(LU_TU2_Y_Name));
    ASSERT_THAT(IdTable, ContainsEntity(LU_TU2_W_Name));

    ASSERT_THAT(IdTable, ContainsEntity(LU_TU1_A_Name));
    ASSERT_THAT(IdTable, ContainsEntity(LU_TU1_B_Name));
    ASSERT_THAT(IdTable, ContainsEntity(LU_TU1_C_Name));
    ASSERT_THAT(IdTable, ContainsEntity(LU_TU2_A_Name));
    ASSERT_THAT(IdTable, ContainsEntity(LU_TU2_B_Name));
    ASSERT_THAT(IdTable, ContainsEntity(LU_TU2_D_Name));

    ASSERT_THAT(IdTable, ContainsEntity(LU_P_Name));
    ASSERT_THAT(IdTable, ContainsEntity(LU_Q_Name));
    ASSERT_THAT(IdTable, ContainsEntity(LU_R_Name));
    ASSERT_THAT(IdTable, ContainsEntity(LU_S_Name));
  }

  // This is safe since we confirmed that these entities are present in the
  // block above.
  const auto LU_TU1_X_Id = Entities.at(LU_TU1_X_Name);
  const auto LU_TU1_Y_Id = Entities.at(LU_TU1_Y_Name);
  const auto LU_TU1_Z_Id = Entities.at(LU_TU1_Z_Name);
  const auto LU_TU2_X_Id = Entities.at(LU_TU2_X_Name);
  const auto LU_TU2_Y_Id = Entities.at(LU_TU2_Y_Name);
  const auto LU_TU2_W_Id = Entities.at(LU_TU2_W_Name);
  const auto LU_TU1_A_Id = Entities.at(LU_TU1_A_Name);
  const auto LU_TU1_B_Id = Entities.at(LU_TU1_B_Name);
  const auto LU_TU1_C_Id = Entities.at(LU_TU1_C_Name);
  const auto LU_TU2_A_Id = Entities.at(LU_TU2_A_Name);
  const auto LU_TU2_B_Id = Entities.at(LU_TU2_B_Name);
  const auto LU_TU2_D_Id = Entities.at(LU_TU2_D_Name);
  const auto LU_P_Id = Entities.at(LU_P_Name);
  const auto LU_Q_Id = Entities.at(LU_Q_Name);
  const auto LU_R_Id = Entities.at(LU_R_Name);
  const auto LU_S_Id = Entities.at(LU_S_Name);

  // LinkageTable Tests.
  {
    ASSERT_THAT(LinkageTable, LinkageTableHasSize(16u));

    ASSERT_THAT(
        LinkageTable,
        EntityHasLinkage(LU_TU1_X_Id, EntityLinkage::LinkageType::None));
    ASSERT_THAT(
        LinkageTable,
        EntityHasLinkage(LU_TU1_Y_Id, EntityLinkage::LinkageType::None));
    ASSERT_THAT(
        LinkageTable,
        EntityHasLinkage(LU_TU1_Z_Id, EntityLinkage::LinkageType::None));
    ASSERT_THAT(
        LinkageTable,
        EntityHasLinkage(LU_TU2_X_Id, EntityLinkage::LinkageType::None));
    ASSERT_THAT(
        LinkageTable,
        EntityHasLinkage(LU_TU2_Y_Id, EntityLinkage::LinkageType::None));
    ASSERT_THAT(
        LinkageTable,
        EntityHasLinkage(LU_TU2_W_Id, EntityLinkage::LinkageType::None));

    ASSERT_THAT(
        LinkageTable,
        EntityHasLinkage(LU_TU1_A_Id, EntityLinkage::LinkageType::Internal));
    ASSERT_THAT(
        LinkageTable,
        EntityHasLinkage(LU_TU1_B_Id, EntityLinkage::LinkageType::Internal));
    ASSERT_THAT(
        LinkageTable,
        EntityHasLinkage(LU_TU1_C_Id, EntityLinkage::LinkageType::Internal));
    ASSERT_THAT(
        LinkageTable,
        EntityHasLinkage(LU_TU2_A_Id, EntityLinkage::LinkageType::Internal));
    ASSERT_THAT(
        LinkageTable,
        EntityHasLinkage(LU_TU2_B_Id, EntityLinkage::LinkageType::Internal));
    ASSERT_THAT(
        LinkageTable,
        EntityHasLinkage(LU_TU2_D_Id, EntityLinkage::LinkageType::Internal));

    ASSERT_THAT(
        LinkageTable,
        EntityHasLinkage(LU_P_Id, EntityLinkage::LinkageType::External));
    ASSERT_THAT(
        LinkageTable,
        EntityHasLinkage(LU_Q_Id, EntityLinkage::LinkageType::External));
    ASSERT_THAT(
        LinkageTable,
        EntityHasLinkage(LU_R_Id, EntityLinkage::LinkageType::External));
    ASSERT_THAT(
        LinkageTable,
        EntityHasLinkage(LU_S_Id, EntityLinkage::LinkageType::External));
  }

  // Data Tests.
  {
    ASSERT_EQ(Data.size(), 2u);

    // Build entity resolution mappings for each TU.
    std::map<EntityId, EntityId> TU1Resolution = {
        {TU1_X_Id, LU_TU1_X_Id}, {TU1_Y_Id, LU_TU1_Y_Id},
        {TU1_Z_Id, LU_TU1_Z_Id}, {TU1_A_Id, LU_TU1_A_Id},
        {TU1_B_Id, LU_TU1_B_Id}, {TU1_C_Id, LU_TU1_C_Id},
        {TU1_P_Id, LU_P_Id},     {TU1_Q_Id, LU_Q_Id},
        {TU1_R_Id, LU_R_Id}};

    std::map<EntityId, EntityId> TU2Resolution = {
        {TU2_X_Id, LU_TU2_X_Id}, {TU2_Y_Id, LU_TU2_Y_Id},
        {TU2_W_Id, LU_TU2_W_Id}, {TU2_A_Id, LU_TU2_A_Id},
        {TU2_B_Id, LU_TU2_B_Id}, {TU2_D_Id, LU_TU2_D_Id},
        {TU2_P_Id, LU_P_Id},     {TU2_Q_Id, LU_Q_Id},
        {TU2_S_Id, LU_S_Id}};

    // S1 Data Tests.
    {
      SummaryName S1("S1");
      ASSERT_NE(Data.find(S1), Data.end());
      const auto &S1Data = Data.at(S1);

      // S1 should contain: TU1(X,Z,A,C,P,R) + TU2(Y,W,B,D,Q,S) = 12 entities.
      ASSERT_THAT(S1Data, SummaryDataHasSize(12u));

      // TU1 entities in S1.
      ASSERT_THAT(S1Data,
                  HasSummaryData(LU_TU1_X_Id, TU1_X_S1_Data, TU1Resolution));
      ASSERT_THAT(S1Data,
                  HasSummaryData(LU_TU1_Z_Id, TU1_Z_S1_Data, TU1Resolution));
      ASSERT_THAT(S1Data,
                  HasSummaryData(LU_TU1_A_Id, TU1_A_S1_Data, TU1Resolution));
      ASSERT_THAT(S1Data,
                  HasSummaryData(LU_TU1_C_Id, TU1_C_S1_Data, TU1Resolution));
      ASSERT_THAT(S1Data,
                  HasSummaryData(LU_P_Id, TU1_P_S1_Data, TU1Resolution));
      ASSERT_THAT(S1Data,
                  HasSummaryData(LU_R_Id, TU1_R_S1_Data, TU1Resolution));

      // TU2 entities in S1.
      ASSERT_THAT(S1Data,
                  HasSummaryData(LU_TU2_Y_Id, TU2_Y_S1_Data, TU2Resolution));
      ASSERT_THAT(S1Data,
                  HasSummaryData(LU_TU2_W_Id, TU2_W_S1_Data, TU2Resolution));
      ASSERT_THAT(S1Data,
                  HasSummaryData(LU_TU2_B_Id, TU2_B_S1_Data, TU2Resolution));
      ASSERT_THAT(S1Data,
                  HasSummaryData(LU_TU2_D_Id, TU2_D_S1_Data, TU2Resolution));
      ASSERT_THAT(S1Data,
                  HasSummaryData(LU_Q_Id, TU2_Q_S1_Data, TU2Resolution));
      ASSERT_THAT(S1Data,
                  HasSummaryData(LU_S_Id, TU2_S_S1_Data, TU2Resolution));
    }

    // S2 Data Tests.
    {
      SummaryName S2("S2");
      ASSERT_NE(Data.find(S2), Data.end());
      const auto &S2Data = Data.at(S2);

      // S2 should contain: TU1(Y,Z,B,C,Q,R) + TU2(X,W,A,D,P,S) = 12 entities.
      ASSERT_THAT(S2Data, SummaryDataHasSize(12u));

      // TU1 entities in S2.
      ASSERT_THAT(S2Data,
                  HasSummaryData(LU_TU1_Y_Id, TU1_Y_S2_Data, TU1Resolution));
      ASSERT_THAT(S2Data,
                  HasSummaryData(LU_TU1_Z_Id, TU1_Z_S2_Data, TU1Resolution));
      ASSERT_THAT(S2Data,
                  HasSummaryData(LU_TU1_B_Id, TU1_B_S2_Data, TU1Resolution));
      ASSERT_THAT(S2Data,
                  HasSummaryData(LU_TU1_C_Id, TU1_C_S2_Data, TU1Resolution));
      ASSERT_THAT(S2Data,
                  HasSummaryData(LU_Q_Id, TU1_Q_S2_Data, TU1Resolution));
      ASSERT_THAT(S2Data,
                  HasSummaryData(LU_R_Id, TU1_R_S2_Data, TU1Resolution));

      // TU2 entities in S2.
      ASSERT_THAT(S2Data,
                  HasSummaryData(LU_TU2_X_Id, TU2_X_S2_Data, TU2Resolution));
      ASSERT_THAT(S2Data,
                  HasSummaryData(LU_TU2_W_Id, TU2_W_S2_Data, TU2Resolution));
      ASSERT_THAT(S2Data,
                  HasSummaryData(LU_TU2_A_Id, TU2_A_S2_Data, TU2Resolution));
      ASSERT_THAT(S2Data,
                  HasSummaryData(LU_TU2_D_Id, TU2_D_S2_Data, TU2Resolution));
      ASSERT_THAT(S2Data,
                  HasSummaryData(LU_P_Id, TU2_P_S2_Data, TU2Resolution));
      ASSERT_THAT(S2Data,
                  HasSummaryData(LU_S_Id, TU2_S_S2_Data, TU2Resolution));
    }
  }
}

// ============================================================================
// Fatal Error Tests
//
// These tests verify that corrupted TU summary data triggers fatal errors.
// Each test constructs a TUSummaryEncoding that violates an invariant by
// directly manipulating internal state via TestFixture accessors, then
// asserts that link() terminates the process.
// ============================================================================

TEST_F(EntityLinkerTest, FatalOnEntityMissingLinkageInformation) {
  // An entity that is in IdTable but has no entry in LinkageTable indicates
  // a corrupted TUSummary and triggers a fatal error.
  NestedBuildNamespace LUNamespace(
      {BuildNamespace(BuildNamespaceKind::LinkUnit, "LU")});
  EntityLinker Linker(LUNamespace);

  auto TU = createTUSummaryEncoding(BuildNamespaceKind::CompilationUnit, "TU");

  // Use addEntity to get a valid EntityId, then remove its linkage entry to
  // simulate a TUSummary where the IdTable and LinkageTable are out of sync.
  const auto Id = addEntity(*TU, "A", EntityLinkage::LinkageType::External);
  getLinkageTable(*TU).erase(Id);

  EXPECT_DEATH(
      { (void)Linker.link(std::move(TU)); },
      "EntityLinker: Entity .* is missing linkage information in TU summary");
}

TEST_F(EntityLinkerTest, FatalOnDuplicateEntityIdInTUSummary) {
  // Two different EntityNames mapping to the same EntityId indicates corrupted
  // TUSummary data and triggers a fatal error.
  NestedBuildNamespace LUNamespace(
      {BuildNamespace(BuildNamespaceKind::LinkUnit, "LU")});
  EntityLinker Linker(LUNamespace);

  auto TU = createTUSummaryEncoding(BuildNamespaceKind::CompilationUnit, "TU");

  // Insert first entity normally.
  const auto Id = addEntity(*TU, "A", EntityLinkage::LinkageType::External);

  // Directly insert a second EntityName that maps to the same EntityId,
  // bypassing the normal getId() uniqueness guarantee.
  NestedBuildNamespace TUNested(getTUNamespace(*TU));
  EntityName SecondName("B", "", TUNested);
  getEntities(getIdTable(*TU)).insert({SecondName, Id});

  EXPECT_DEATH(
      { (void)Linker.link(std::move(TU)); },
      "EntityLinker: Duplicate entity ID .* in TU summary");
}

TEST_F(EntityLinkerTest, FatalOnEntityNotFoundInResolutionTable) {
  // Summary data that references an EntityId not present in
  // IdTable/LinkageTable will not appear in the resolution table, triggering a
  // fatal error in merge.
  NestedBuildNamespace LUNamespace(
      {BuildNamespace(BuildNamespaceKind::LinkUnit, "LU")});
  EntityLinker Linker(LUNamespace);

  auto TU = createTUSummaryEncoding(BuildNamespaceKind::CompilationUnit, "TU");

  // Register one entity normally so resolution succeeds for it.
  addEntity(*TU, "A", EntityLinkage::LinkageType::External);

  // Obtain a second EntityId from a separate table — it will never appear in
  // the TU's resolution table because it is not in TU's IdTable or
  // LinkageTable.
  EntityIdTable AuxTable;
  NestedBuildNamespace TUNested(getTUNamespace(*TU));
  EntityName AuxName("B", "", TUNested);
  const auto OrphanId = AuxTable.getId(AuxName);

  // Insert summary data keyed on the orphan ID.
  SummaryName SN("S1");
  getData(*TU)[SN][OrphanId] = std::make_unique<MockEntitySummaryEncoding>();

  EXPECT_DEATH(
      { (void)Linker.link(std::move(TU)); },
      "EntityLinker: Entity .* not found in EntityResolutionTable");
}

TEST_F(EntityLinkerTest, FatalOnEntityAlreadyExistsInLinkageTableForNone) {
  // If the output LinkageTable already contains the LU EntityId that a None
  // linkage entity resolves to, a fatal error is triggered. This can only
  // happen due to data corruption or a bug in resolve logic, so we simulate
  // it by pre-populating the output.
  NestedBuildNamespace LUNamespace(
      {BuildNamespace(BuildNamespaceKind::LinkUnit, "LU")});
  EntityLinker Linker(LUNamespace);

  // Link a first TU so the output already has an entry for entity "A"/None.
  auto TU1 = createTUSummaryEncoding(BuildNamespaceKind::CompilationUnit, "TU");
  addEntity(*TU1, "A", EntityLinkage::LinkageType::None);
  const BuildNamespace TUNamespace = getTUNamespace(*TU1);
  ASSERT_THAT_ERROR(Linker.link(std::move(TU1)), llvm::Succeeded());

  // Determine the LU EntityId that was assigned to A.
  NestedBuildNamespace LocalNS =
      NestedBuildNamespace(TUNamespace).makeQualified(LUNamespace);
  EntityName LU_A_Name("A", "", LocalNS);
  const auto LU_A_Id =
      getEntities(getIdTable(Linker.getOutput())).at(LU_A_Name);

  // Inject the resolved LU name -> LU_A_Id mapping into the output IdTable so
  // that getId() returns LU_A_Id when TU2's "A"/None entity is resolved. Since
  // LU_A_Id is already in OutLinkage (from TU1), the try_emplace in
  // resolveEntity() will fail, triggering the fatal.
  auto &OutEntities = getEntities(getIdTable(Linker.getOutput()));
  auto &OutLinkage = getLinkageTable(Linker.getOutput());

  // Inject a second name -> same id mapping in the output IdTable so the next
  // getId() call returns LU_A_Id for a new name that a fresh TU entity resolves
  // to — and the LinkageTable already has that id, triggering the fatal.
  NestedBuildNamespace FakeTUNested(
      BuildNamespace(BuildNamespaceKind::CompilationUnit, "TU2"));
  NestedBuildNamespace FakeLocalNS = FakeTUNested.makeQualified(LUNamespace);
  EntityName FakeA("A", "", FakeLocalNS);
  OutEntities.insert({FakeA, LU_A_Id});
  // LU_A_Id is already in OutLinkage from linking TU1 with None linkage.
  ASSERT_NE(OutLinkage.find(LU_A_Id), OutLinkage.end());

  // Now link a TU2 whose "A" entity resolves to FakeLocalNS, which maps to
  // LU_A_Id (already in LinkageTable) — triggering the fatal.
  auto TU2 =
      createTUSummaryEncoding(BuildNamespaceKind::CompilationUnit, "TU2");
  addEntity(*TU2, "A", EntityLinkage::LinkageType::None);

  EXPECT_DEATH(
      { (void)Linker.link(std::move(TU2)); },
      "EntityLinker: Entity .* with .* linkage already exists in LinkageTable");
}

TEST_F(EntityLinkerTest, FatalOnInternalEntityAlreadyExistsInLinkageTable) {
  // Same as above but for Internal linkage.
  NestedBuildNamespace LUNamespace(
      {BuildNamespace(BuildNamespaceKind::LinkUnit, "LU")});
  EntityLinker Linker(LUNamespace);

  auto TU1 = createTUSummaryEncoding(BuildNamespaceKind::CompilationUnit, "TU");
  addEntity(*TU1, "A", EntityLinkage::LinkageType::Internal);
  const BuildNamespace TUNamespace = getTUNamespace(*TU1);
  ASSERT_THAT_ERROR(Linker.link(std::move(TU1)), llvm::Succeeded());

  NestedBuildNamespace LocalNS =
      NestedBuildNamespace(TUNamespace).makeQualified(LUNamespace);
  EntityName LU_A_Name("A", "", LocalNS);
  const auto LU_A_Id =
      getEntities(getIdTable(Linker.getOutput())).at(LU_A_Name);

  auto &OutEntities = getEntities(getIdTable(Linker.getOutput()));
  auto &OutLinkage = getLinkageTable(Linker.getOutput());

  NestedBuildNamespace FakeTUNested(
      BuildNamespace(BuildNamespaceKind::CompilationUnit, "TU2"));
  NestedBuildNamespace FakeLocalNS = FakeTUNested.makeQualified(LUNamespace);
  EntityName FakeA("A", "", FakeLocalNS);
  OutEntities.insert({FakeA, LU_A_Id});
  ASSERT_NE(OutLinkage.find(LU_A_Id), OutLinkage.end());

  auto TU2 =
      createTUSummaryEncoding(BuildNamespaceKind::CompilationUnit, "TU2");
  addEntity(*TU2, "A", EntityLinkage::LinkageType::Internal);

  EXPECT_DEATH(
      { (void)Linker.link(std::move(TU2)); },
      "EntityLinker: Entity .* with .* linkage already exists in LinkageTable");
}

TEST_F(EntityLinkerTest,
       FatalOnFailedInsertNoneLinkageEntityIntoOutputSummary) {
  // If a None linkage entity's summary data cannot be inserted into the output
  // (because its LU EntityId is already present for the same SummaryName), a
  // fatal error is triggered. We simulate this by pre-populating the output's
  // data map with the target LU EntityId while keeping that ID absent from
  // OutLinkage, so resolveEntity() succeeds (first insertion into LinkageTable)
  // but merge() then finds the data slot already occupied.
  NestedBuildNamespace LUNamespace(
      {BuildNamespace(BuildNamespaceKind::LinkUnit, "LU")});
  EntityLinker Linker(LUNamespace);

  // Compute the LU namespace that TU2's None entity "A" will resolve to.
  BuildNamespace TU2NS(BuildNamespaceKind::CompilationUnit, "TU2");
  NestedBuildNamespace TU2LocalNS =
      NestedBuildNamespace(TU2NS).makeQualified(LUNamespace);
  EntityName LU_A_Name("A", "", TU2LocalNS);

  // Allocate the LU EntityId for A by registering the name in the output
  // IdTable, then pre-insert summary data for it — but do NOT insert into
  // LinkageTable, so resolveEntity's try_emplace will succeed.
  const auto LU_A_Id = getIdTable(Linker.getOutput()).getId(LU_A_Name);
  SummaryName SN("S1");
  getData(Linker.getOutput())[SN].try_emplace(
      LU_A_Id, std::make_unique<MockEntitySummaryEncoding>());

  // Link TU2: its "A"/None entity resolves to LU_A_Id (already in IdTable),
  // resolveEntity inserts it into LinkageTable (succeeds), then merge fails
  // because Data[S1][LU_A_Id] is already occupied.
  auto TU2 =
      createTUSummaryEncoding(BuildNamespaceKind::CompilationUnit, "TU2");
  const auto TU2_A_Id = addEntity(*TU2, "A", EntityLinkage::LinkageType::None);
  addSummaryData(*TU2, TU2_A_Id, "S1");

  EXPECT_DEATH(
      { (void)Linker.link(std::move(TU2)); },
      "EntityLinker: Failed to insert data against SummaryName");
}

} // namespace
