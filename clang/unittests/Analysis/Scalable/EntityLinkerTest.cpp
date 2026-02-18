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

namespace clang::ssaf {

namespace {

// Mock EntitySummaryEncoding for testing
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
  // Helper to create a TUSummaryEncoding with entities
  std::unique_ptr<TUSummaryEncoding>
  createTUSummaryEncoding(BuildNamespaceKind Kind, llvm::StringRef Name) {
    return std::make_unique<TUSummaryEncoding>(BuildNamespace(Kind, Name));
  }

  // Helper to add an entity to a TUSummaryEncoding
  EntityId addEntity(TUSummaryEncoding &TU, llvm::StringRef USR,
                     EntityLinkage::LinkageType Linkage) {
    EntityName Name(USR, "", NestedBuildNamespace(getTUNamespace(TU)));
    EntityId Id = getIdTable(TU).getId(Name);
    getLinkageTable(TU).insert({Id, EntityLinkage(Linkage)});
    return Id;
  }

  // Helper to add summary data to a TUSummaryEncoding
  size_t addSummaryData(TUSummaryEncoding &TU, EntityId EId,
                        llvm::StringRef SummaryNameStr) {
    SummaryName SN(SummaryNameStr.str());
    auto Summary = std::make_unique<MockEntitySummaryEncoding>();
    const size_t ESId = Summary->getId();
    getData(TU)[SN][EId] = std::move(Summary);
    return ESId;
  }
};

TEST_F(EntityLinkerTest, NoLink) {
  NestedBuildNamespace LUNamespace(
      {BuildNamespace(BuildNamespaceKind::LinkUnit, "LU")});

  EntityLinker Linker(LUNamespace);

  const auto &Output = Linker.getOutput();
  EXPECT_EQ(getIdTable(Output).count(), 0u);
  EXPECT_EQ(getLinkageTable(Output).size(), 0u);
  EXPECT_EQ(getData(Output).size(), 0u);
}

TEST_F(EntityLinkerTest, EmptyLink) {
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

TEST_F(EntityLinkerTest, NonEmptyLink) {
  NestedBuildNamespace LUNamespace(
      {BuildNamespace(BuildNamespaceKind::LinkUnit, "LU")});

  EntityLinker Linker(LUNamespace);

  auto TU = createTUSummaryEncoding(BuildNamespaceKind::CompilationUnit, "TU");

  const auto EIdA = addEntity(*TU, "A", EntityLinkage::LinkageType::None);
  const auto ESIdAS1 = addSummaryData(*TU, EIdA, "S1");
  const auto ESIdAS2 = addSummaryData(*TU, EIdA, "S2");

  const auto EIdB = addEntity(*TU, "B", EntityLinkage::LinkageType::Internal);
  const auto ESIdBS1 = addSummaryData(*TU, EIdB, "S1");
  const auto ESIdBS2 = addSummaryData(*TU, EIdB, "S2");

  const auto EIdC = addEntity(*TU, "C", EntityLinkage::LinkageType::External);
  const auto ESIdCS1 = addSummaryData(*TU, EIdC, "S1");

  const auto EIdD = addEntity(*TU, "D", EntityLinkage::LinkageType::External);
  const auto ESIdDS2 = addSummaryData(*TU, EIdD, "S2");

  const BuildNamespace TUNamespace = getTUNamespace(*TU);

  EXPECT_THAT_ERROR(Linker.link(std::move(TU)), llvm::Succeeded());

  const auto &Output = Linker.getOutput();
  const auto &IdTable = getIdTable(Output);
  const auto &LinkageTable = getLinkageTable(Output);
  const auto &Data = getData(Output);

  // Construct the nested namespace with TU inside LU
  std::vector<BuildNamespace> NamespaceVec;
  NamespaceVec.push_back(TUNamespace);
  for (const auto &NS : getNamespaces(LUNamespace)) {
    NamespaceVec.push_back(NS);
  }
  NestedBuildNamespace LocalNamespace(NamespaceVec);

  EntityName NameA("A", "", LocalNamespace);
  EntityName NameB("B", "", LocalNamespace);
  EntityName NameC("C", "", LUNamespace);
  EntityName NameD("D", "", LUNamespace);

  // EntityIDTable Tests.
  {
    const auto &Entities = getEntities(IdTable);

    EXPECT_EQ(IdTable.count(), 4u);

    EXPECT_TRUE(IdTable.contains(NameA));
    EXPECT_EQ(Entities.at(NameA), EIdA);

    EXPECT_TRUE(IdTable.contains(NameB));
    EXPECT_EQ(Entities.at(NameB), EIdB);

    EXPECT_TRUE(IdTable.contains(NameC));
    EXPECT_EQ(Entities.at(NameC), EIdC);

    EXPECT_TRUE(IdTable.contains(NameD));
    EXPECT_EQ(Entities.at(NameD), EIdD);
  }

  // LinkageTable Tests.
  {
    EXPECT_EQ(LinkageTable.size(), 4u);

    ASSERT_NE(LinkageTable.find(EIdA), LinkageTable.end());
    EXPECT_EQ(getLinkage(LinkageTable.at(EIdA)),
              EntityLinkage::LinkageType::None);

    ASSERT_NE(LinkageTable.find(EIdB), LinkageTable.end());
    EXPECT_EQ(LinkageTable.at(EIdB).getLinkage(),
              EntityLinkage::LinkageType::Internal);

    ASSERT_NE(LinkageTable.find(EIdC), LinkageTable.end());
    EXPECT_EQ(LinkageTable.at(EIdC).getLinkage(),
              EntityLinkage::LinkageType::External);

    ASSERT_NE(LinkageTable.find(EIdD), LinkageTable.end());
    EXPECT_EQ(LinkageTable.at(EIdD).getLinkage(),
              EntityLinkage::LinkageType::External);
  }

  // Data Tests.
  {
    EXPECT_EQ(Data.size(), 2u);

    std::map<EntityId, EntityId> ExpectedEntityResolutionMapping = {
        {EIdA, EIdA}, {EIdB, EIdB}, {EIdC, EIdC}, {EIdD, EIdD}};

    // S1 Tests.
    {
      SummaryName S1("S1");
      ASSERT_NE(Data.find(S1), Data.end());

      const auto &S1Data = Data.at(S1);
      EXPECT_EQ(S1Data.size(), 3u);

      EXPECT_NE(S1Data.find(EIdA), S1Data.end());
      auto *MockA =
          static_cast<MockEntitySummaryEncoding *>(S1Data.at(EIdA).get());
      EXPECT_EQ(MockA->getId(), ESIdAS1);
      EXPECT_EQ(MockA->getPatchedIds(), ExpectedEntityResolutionMapping);

      EXPECT_NE(S1Data.find(EIdB), S1Data.end());
      auto *MockB =
          static_cast<MockEntitySummaryEncoding *>(S1Data.at(EIdB).get());
      EXPECT_EQ(MockB->getId(), ESIdBS1);
      EXPECT_EQ(MockB->getPatchedIds(), ExpectedEntityResolutionMapping);

      EXPECT_NE(S1Data.find(EIdC), S1Data.end());
      auto *MockC =
          static_cast<MockEntitySummaryEncoding *>(S1Data.at(EIdC).get());
      EXPECT_EQ(MockC->getId(), ESIdCS1);
      EXPECT_EQ(MockC->getPatchedIds(), ExpectedEntityResolutionMapping);
    }

    // S2 Tests.
    {
      SummaryName S2("S2");
      ASSERT_NE(Data.find(S2), Data.end());

      const auto &S2Data = Data.at(S2);
      EXPECT_EQ(S2Data.size(), 3u);

      EXPECT_NE(S2Data.find(EIdA), S2Data.end());
      auto *MockA =
          static_cast<MockEntitySummaryEncoding *>(S2Data.at(EIdA).get());
      EXPECT_EQ(MockA->getId(), ESIdAS2);
      EXPECT_EQ(MockA->getPatchedIds(), ExpectedEntityResolutionMapping);

      EXPECT_NE(S2Data.find(EIdB), S2Data.end());
      auto *MockB =
          static_cast<MockEntitySummaryEncoding *>(S2Data.at(EIdB).get());
      EXPECT_EQ(MockB->getId(), ESIdBS2);
      EXPECT_EQ(MockB->getPatchedIds(), ExpectedEntityResolutionMapping);

      EXPECT_NE(S2Data.find(EIdD), S2Data.end());
      auto *MockD =
          static_cast<MockEntitySummaryEncoding *>(S2Data.at(EIdD).get());
      EXPECT_EQ(MockD->getId(), ESIdDS2);
      EXPECT_EQ(MockD->getPatchedIds(), ExpectedEntityResolutionMapping);
    }
  }
}

TEST_F(EntityLinkerTest, TwoTULinkWithAllCombinations) {
  NestedBuildNamespace LUNamespace(
      {BuildNamespace(BuildNamespaceKind::LinkUnit, "LU")});

  EntityLinker Linker(LUNamespace);

  // Create TU1 with entities covering all linkage types and summary
  // distributions
  auto TU1 =
      createTUSummaryEncoding(BuildNamespaceKind::CompilationUnit, "TU1");

  // None linkage entities in TU1
  const auto EIdTU1_X_None =
      addEntity(*TU1, "X", EntityLinkage::LinkageType::None);
  const auto ESIdTU1_X_S1 = addSummaryData(*TU1, EIdTU1_X_None, "S1");

  const auto EIdTU1_Y_None =
      addEntity(*TU1, "Y", EntityLinkage::LinkageType::None);
  const auto ESIdTU1_Y_S2 = addSummaryData(*TU1, EIdTU1_Y_None, "S2");

  const auto EIdTU1_Z_None =
      addEntity(*TU1, "Z", EntityLinkage::LinkageType::None);
  const auto ESIdTU1_Z_S1 = addSummaryData(*TU1, EIdTU1_Z_None, "S1");
  const auto ESIdTU1_Z_S2 = addSummaryData(*TU1, EIdTU1_Z_None, "S2");

  // Internal linkage entities in TU1
  const auto EIdTU1_A_Internal =
      addEntity(*TU1, "A", EntityLinkage::LinkageType::Internal);
  const auto ESIdTU1_A_S1 = addSummaryData(*TU1, EIdTU1_A_Internal, "S1");

  const auto EIdTU1_B_Internal =
      addEntity(*TU1, "B", EntityLinkage::LinkageType::Internal);
  const auto ESIdTU1_B_S2 = addSummaryData(*TU1, EIdTU1_B_Internal, "S2");

  const auto EIdTU1_C_Internal =
      addEntity(*TU1, "C", EntityLinkage::LinkageType::Internal);
  const auto ESIdTU1_C_S1 = addSummaryData(*TU1, EIdTU1_C_Internal, "S1");
  const auto ESIdTU1_C_S2 = addSummaryData(*TU1, EIdTU1_C_Internal, "S2");

  // External linkage entities in TU1
  const auto EIdTU1_P_External =
      addEntity(*TU1, "P", EntityLinkage::LinkageType::External);
  const auto ESIdTU1_P_S1 = addSummaryData(*TU1, EIdTU1_P_External, "S1");

  const auto EIdTU1_Q_External =
      addEntity(*TU1, "Q", EntityLinkage::LinkageType::External);
  const auto ESIdTU1_Q_S2 = addSummaryData(*TU1, EIdTU1_Q_External, "S2");

  const auto EIdTU1_R_External =
      addEntity(*TU1, "R", EntityLinkage::LinkageType::External);
  const auto ESIdTU1_R_S1 = addSummaryData(*TU1, EIdTU1_R_External, "S1");
  const auto ESIdTU1_R_S2 = addSummaryData(*TU1, EIdTU1_R_External, "S2");

  const BuildNamespace TU1Namespace = getTUNamespace(*TU1);

  // Link TU1
  EXPECT_THAT_ERROR(Linker.link(std::move(TU1)), llvm::Succeeded());

  // Create TU2 with entities covering all combinations including duplicates
  auto TU2 =
      createTUSummaryEncoding(BuildNamespaceKind::CompilationUnit, "TU2");

  // None linkage entities in TU2 - includes duplicates and unique
  const auto EIdTU2_X_None =
      addEntity(*TU2, "X", EntityLinkage::LinkageType::None);
  const auto ESIdTU2_X_S2 = addSummaryData(*TU2, EIdTU2_X_None, "S2");

  const auto EIdTU2_Y_None =
      addEntity(*TU2, "Y", EntityLinkage::LinkageType::None);
  const auto ESIdTU2_Y_S1 = addSummaryData(*TU2, EIdTU2_Y_None, "S1");

  const auto EIdTU2_W_None =
      addEntity(*TU2, "W", EntityLinkage::LinkageType::None);
  const auto ESIdTU2_W_S1 = addSummaryData(*TU2, EIdTU2_W_None, "S1");
  const auto ESIdTU2_W_S2 = addSummaryData(*TU2, EIdTU2_W_None, "S2");

  // Internal linkage entities in TU2 - includes duplicates and unique
  const auto EIdTU2_A_Internal =
      addEntity(*TU2, "A", EntityLinkage::LinkageType::Internal);
  const auto ESIdTU2_A_S2 = addSummaryData(*TU2, EIdTU2_A_Internal, "S2");

  const auto EIdTU2_B_Internal =
      addEntity(*TU2, "B", EntityLinkage::LinkageType::Internal);
  const auto ESIdTU2_B_S1 = addSummaryData(*TU2, EIdTU2_B_Internal, "S1");

  const auto EIdTU2_D_Internal =
      addEntity(*TU2, "D", EntityLinkage::LinkageType::Internal);
  const auto ESIdTU2_D_S1 = addSummaryData(*TU2, EIdTU2_D_Internal, "S1");
  const auto ESIdTU2_D_S2 = addSummaryData(*TU2, EIdTU2_D_Internal, "S2");

  // External linkage entities in TU2 - includes duplicates (will be dropped)
  // and unique
  const auto EIdTU2_P_External =
      addEntity(*TU2, "P", EntityLinkage::LinkageType::External);
  const auto ESIdTU2_P_S2 = addSummaryData(*TU2, EIdTU2_P_External, "S2");

  const auto EIdTU2_Q_External =
      addEntity(*TU2, "Q", EntityLinkage::LinkageType::External);
  const auto ESIdTU2_Q_S1 = addSummaryData(*TU2, EIdTU2_Q_External, "S1");

  const auto EIdTU2_S_External =
      addEntity(*TU2, "S", EntityLinkage::LinkageType::External);
  const auto ESIdTU2_S_S1 = addSummaryData(*TU2, EIdTU2_S_External, "S1");
  const auto ESIdTU2_S_S2 = addSummaryData(*TU2, EIdTU2_S_External, "S2");

  const BuildNamespace TU2Namespace = getTUNamespace(*TU2);

  // Link TU2
  EXPECT_THAT_ERROR(Linker.link(std::move(TU2)), llvm::Succeeded());

  // Verify the output
  const auto &Output = Linker.getOutput();
  const auto &IdTable = getIdTable(Output);
  const auto &LinkageTable = getLinkageTable(Output);
  const auto &Data = getData(Output);

  // Construct the nested namespaces
  std::vector<BuildNamespace> TU1NamespaceVec;
  TU1NamespaceVec.push_back(TU1Namespace);
  for (const auto &NS : getNamespaces(LUNamespace)) {
    TU1NamespaceVec.push_back(NS);
  }
  NestedBuildNamespace TU1LocalNamespace(TU1NamespaceVec);

  std::vector<BuildNamespace> TU2NamespaceVec;
  TU2NamespaceVec.push_back(TU2Namespace);
  for (const auto &NS : getNamespaces(LUNamespace)) {
    TU2NamespaceVec.push_back(NS);
  }
  NestedBuildNamespace TU2LocalNamespace(TU2NamespaceVec);

  // Create expected entity names
  // None linkage entities use local namespace (TU scoped)
  EntityName NameTU1_X_None("X", "", TU1LocalNamespace);
  EntityName NameTU1_Y_None("Y", "", TU1LocalNamespace);
  EntityName NameTU1_Z_None("Z", "", TU1LocalNamespace);
  EntityName NameTU2_X_None("X", "", TU2LocalNamespace);
  EntityName NameTU2_Y_None("Y", "", TU2LocalNamespace);
  EntityName NameTU2_W_None("W", "", TU2LocalNamespace);

  // Internal linkage entities use local namespace (TU scoped)
  EntityName NameTU1_A_Internal("A", "", TU1LocalNamespace);
  EntityName NameTU1_B_Internal("B", "", TU1LocalNamespace);
  EntityName NameTU1_C_Internal("C", "", TU1LocalNamespace);
  EntityName NameTU2_A_Internal("A", "", TU2LocalNamespace);
  EntityName NameTU2_B_Internal("B", "", TU2LocalNamespace);
  EntityName NameTU2_D_Internal("D", "", TU2LocalNamespace);

  // External linkage entities use LU namespace (shared across TUs)
  EntityName NameP_External("P", "", LUNamespace);
  EntityName NameQ_External("Q", "", LUNamespace);
  EntityName NameR_External("R", "", LUNamespace);
  EntityName NameS_External("S", "", LUNamespace);

  // EntityIdTable Tests
  {
    const auto &Entities = getEntities(IdTable);

    // Should have 6 None + 6 Internal + 4 External = 16 entities total
    EXPECT_EQ(IdTable.count(), 16u);

    // TU1 None linkage entities
    EXPECT_TRUE(IdTable.contains(NameTU1_X_None));
    ASSERT_EQ(Entities.at(NameTU1_X_None), EIdTU1_X_None);

    EXPECT_TRUE(IdTable.contains(NameTU1_Y_None));
    EXPECT_EQ(Entities.at(NameTU1_Y_None), EIdTU1_Y_None);

    EXPECT_TRUE(IdTable.contains(NameTU1_Z_None));
    EXPECT_EQ(Entities.at(NameTU1_Z_None), EIdTU1_Z_None);

    // TU2 None linkage entities (different from TU1 due to namespace)
    EXPECT_TRUE(IdTable.contains(NameTU2_X_None));
    EXPECT_EQ(Entities.at(NameTU2_X_None), EIdTU2_X_None);

    EXPECT_TRUE(IdTable.contains(NameTU2_Y_None));
    EXPECT_EQ(Entities.at(NameTU2_Y_None), EIdTU2_Y_None);

    EXPECT_TRUE(IdTable.contains(NameTU2_W_None));
    EXPECT_EQ(Entities.at(NameTU2_W_None), EIdTU2_W_None);

    // TU1 Internal linkage entities
    EXPECT_TRUE(IdTable.contains(NameTU1_A_Internal));
    EXPECT_EQ(Entities.at(NameTU1_A_Internal), EIdTU1_A_Internal);

    EXPECT_TRUE(IdTable.contains(NameTU1_B_Internal));
    EXPECT_EQ(Entities.at(NameTU1_B_Internal), EIdTU1_B_Internal);

    EXPECT_TRUE(IdTable.contains(NameTU1_C_Internal));
    EXPECT_EQ(Entities.at(NameTU1_C_Internal), EIdTU1_C_Internal);

    // TU2 Internal linkage entities (different from TU1 due to namespace)
    EXPECT_TRUE(IdTable.contains(NameTU2_A_Internal));
    EXPECT_EQ(Entities.at(NameTU2_A_Internal), EIdTU2_A_Internal);

    EXPECT_TRUE(IdTable.contains(NameTU2_B_Internal));
    EXPECT_EQ(Entities.at(NameTU2_B_Internal), EIdTU2_B_Internal);

    EXPECT_TRUE(IdTable.contains(NameTU2_D_Internal));
    EXPECT_EQ(Entities.at(NameTU2_D_Internal), EIdTU2_D_Internal);

    // External linkage entities (shared across TUs)
    EXPECT_TRUE(IdTable.contains(NameP_External));
    EXPECT_EQ(Entities.at(NameP_External), EIdTU1_P_External);

    EXPECT_TRUE(IdTable.contains(NameQ_External));
    EXPECT_EQ(Entities.at(NameQ_External), EIdTU1_Q_External);

    EXPECT_TRUE(IdTable.contains(NameR_External));
    EXPECT_EQ(Entities.at(NameR_External), EIdTU1_R_External);

    EXPECT_TRUE(IdTable.contains(NameS_External));
    EXPECT_EQ(Entities.at(NameS_External), EIdTU2_S_External);
  }

  // LinkageTable Tests
  {
    EXPECT_EQ(LinkageTable.size(), 16u);

    // Verify None linkage entities
    EXPECT_EQ(LinkageTable.at(EIdTU1_X_None).getLinkage(),
              EntityLinkage::LinkageType::None);
    EXPECT_EQ(LinkageTable.at(EIdTU1_Y_None).getLinkage(),
              EntityLinkage::LinkageType::None);
    EXPECT_EQ(LinkageTable.at(EIdTU1_Z_None).getLinkage(),
              EntityLinkage::LinkageType::None);
    EXPECT_EQ(LinkageTable.at(EIdTU2_X_None).getLinkage(),
              EntityLinkage::LinkageType::None);
    EXPECT_EQ(LinkageTable.at(EIdTU2_Y_None).getLinkage(),
              EntityLinkage::LinkageType::None);
    EXPECT_EQ(LinkageTable.at(EIdTU2_W_None).getLinkage(),
              EntityLinkage::LinkageType::None);

    // Verify Internal linkage entities
    EXPECT_EQ(LinkageTable.at(EIdTU1_A_Internal).getLinkage(),
              EntityLinkage::LinkageType::Internal);
    EXPECT_EQ(LinkageTable.at(EIdTU1_B_Internal).getLinkage(),
              EntityLinkage::LinkageType::Internal);
    EXPECT_EQ(LinkageTable.at(EIdTU1_C_Internal).getLinkage(),
              EntityLinkage::LinkageType::Internal);
    EXPECT_EQ(LinkageTable.at(EIdTU2_A_Internal).getLinkage(),
              EntityLinkage::LinkageType::Internal);
    EXPECT_EQ(LinkageTable.at(EIdTU2_B_Internal).getLinkage(),
              EntityLinkage::LinkageType::Internal);
    EXPECT_EQ(LinkageTable.at(EIdTU2_D_Internal).getLinkage(),
              EntityLinkage::LinkageType::Internal);

    // Verify External linkage entities
    EXPECT_EQ(LinkageTable.at(EIdTU1_P_External).getLinkage(),
              EntityLinkage::LinkageType::External);
    EXPECT_EQ(LinkageTable.at(EIdTU1_Q_External).getLinkage(),
              EntityLinkage::LinkageType::External);
    EXPECT_EQ(LinkageTable.at(EIdTU1_R_External).getLinkage(),
              EntityLinkage::LinkageType::External);
    EXPECT_EQ(LinkageTable.at(EIdTU2_S_External).getLinkage(),
              EntityLinkage::LinkageType::External);
  }

  // Data Tests
  {
    EXPECT_EQ(Data.size(), 2u);

    // Build entity resolution mappings for each TU
    std::map<EntityId, EntityId> TU1EntityResolutionMapping = {
        {EIdTU1_X_None, EIdTU1_X_None},
        {EIdTU1_Y_None, EIdTU1_Y_None},
        {EIdTU1_Z_None, EIdTU1_Z_None},
        {EIdTU1_A_Internal, EIdTU1_A_Internal},
        {EIdTU1_B_Internal, EIdTU1_B_Internal},
        {EIdTU1_C_Internal, EIdTU1_C_Internal},
        {EIdTU1_P_External, EIdTU1_P_External},
        {EIdTU1_Q_External, EIdTU1_Q_External},
        {EIdTU1_R_External, EIdTU1_R_External}};

    std::map<EntityId, EntityId> TU2EntityResolutionMapping = {
        {EIdTU2_X_None, EIdTU2_X_None},
        {EIdTU2_Y_None, EIdTU2_Y_None},
        {EIdTU2_W_None, EIdTU2_W_None},
        {EIdTU2_A_Internal, EIdTU2_A_Internal},
        {EIdTU2_B_Internal, EIdTU2_B_Internal},
        {EIdTU2_D_Internal, EIdTU2_D_Internal},
        // External linkage entities from TU2 resolve to TU1's IDs if duplicate
        {EIdTU2_P_External, EIdTU1_P_External},
        {EIdTU2_Q_External, EIdTU1_Q_External},
        {EIdTU2_S_External, EIdTU2_S_External}};

    // S1 Tests
    {
      SummaryName S1("S1");
      ASSERT_NE(Data.find(S1), Data.end());

      const auto &S1Data = Data.at(S1);
      // S1 should contain: TU1(X,Z,A,C,P,R) + TU2(Y,W,B,D,S) = 11 entities
      // Note: TU2's P and Q external entities are dropped because TU1 already
      // has them
      EXPECT_EQ(S1Data.size(), 11u);

      // Verify TU1 entities in S1
      EXPECT_NE(S1Data.find(EIdTU1_X_None), S1Data.end());
      auto *MockTU1_X = static_cast<MockEntitySummaryEncoding *>(
          S1Data.at(EIdTU1_X_None).get());
      EXPECT_EQ(MockTU1_X->getId(), ESIdTU1_X_S1);
      EXPECT_EQ(MockTU1_X->getPatchedIds(), TU1EntityResolutionMapping);

      EXPECT_NE(S1Data.find(EIdTU1_Z_None), S1Data.end());
      auto *MockTU1_Z = static_cast<MockEntitySummaryEncoding *>(
          S1Data.at(EIdTU1_Z_None).get());
      EXPECT_EQ(MockTU1_Z->getId(), ESIdTU1_Z_S1);
      EXPECT_EQ(MockTU1_Z->getPatchedIds(), TU1EntityResolutionMapping);

      EXPECT_NE(S1Data.find(EIdTU1_A_Internal), S1Data.end());
      auto *MockTU1_A = static_cast<MockEntitySummaryEncoding *>(
          S1Data.at(EIdTU1_A_Internal).get());
      EXPECT_EQ(MockTU1_A->getId(), ESIdTU1_A_S1);
      EXPECT_EQ(MockTU1_A->getPatchedIds(), TU1EntityResolutionMapping);

      EXPECT_NE(S1Data.find(EIdTU1_C_Internal), S1Data.end());
      auto *MockTU1_C = static_cast<MockEntitySummaryEncoding *>(
          S1Data.at(EIdTU1_C_Internal).get());
      EXPECT_EQ(MockTU1_C->getId(), ESIdTU1_C_S1);
      EXPECT_EQ(MockTU1_C->getPatchedIds(), TU1EntityResolutionMapping);

      EXPECT_NE(S1Data.find(EIdTU1_P_External), S1Data.end());
      auto *MockTU1_P = static_cast<MockEntitySummaryEncoding *>(
          S1Data.at(EIdTU1_P_External).get());
      EXPECT_EQ(MockTU1_P->getId(), ESIdTU1_P_S1);
      EXPECT_EQ(MockTU1_P->getPatchedIds(), TU1EntityResolutionMapping);

      EXPECT_NE(S1Data.find(EIdTU1_R_External), S1Data.end());
      auto *MockTU1_R = static_cast<MockEntitySummaryEncoding *>(
          S1Data.at(EIdTU1_R_External).get());
      EXPECT_EQ(MockTU1_R->getId(), ESIdTU1_R_S1);
      EXPECT_EQ(MockTU1_R->getPatchedIds(), TU1EntityResolutionMapping);

      // Verify TU2 entities in S1
      EXPECT_NE(S1Data.find(EIdTU2_Y_None), S1Data.end());
      auto *MockTU2_Y = static_cast<MockEntitySummaryEncoding *>(
          S1Data.at(EIdTU2_Y_None).get());
      EXPECT_EQ(MockTU2_Y->getId(), ESIdTU2_Y_S1);
      EXPECT_EQ(MockTU2_Y->getPatchedIds(), TU2EntityResolutionMapping);

      EXPECT_NE(S1Data.find(EIdTU2_W_None), S1Data.end());
      auto *MockTU2_W = static_cast<MockEntitySummaryEncoding *>(
          S1Data.at(EIdTU2_W_None).get());
      EXPECT_EQ(MockTU2_W->getId(), ESIdTU2_W_S1);
      EXPECT_EQ(MockTU2_W->getPatchedIds(), TU2EntityResolutionMapping);

      EXPECT_NE(S1Data.find(EIdTU2_B_Internal), S1Data.end());
      auto *MockTU2_B = static_cast<MockEntitySummaryEncoding *>(
          S1Data.at(EIdTU2_B_Internal).get());
      EXPECT_EQ(MockTU2_B->getId(), ESIdTU2_B_S1);
      EXPECT_EQ(MockTU2_B->getPatchedIds(), TU2EntityResolutionMapping);

      EXPECT_NE(S1Data.find(EIdTU2_D_Internal), S1Data.end());
      auto *MockTU2_D = static_cast<MockEntitySummaryEncoding *>(
          S1Data.at(EIdTU2_D_Internal).get());
      EXPECT_EQ(MockTU2_D->getId(), ESIdTU2_D_S1);
      EXPECT_EQ(MockTU2_D->getPatchedIds(), TU2EntityResolutionMapping);

      EXPECT_NE(S1Data.find(EIdTU2_S_External), S1Data.end());
      auto *MockTU2_S = static_cast<MockEntitySummaryEncoding *>(
          S1Data.at(EIdTU2_S_External).get());
      EXPECT_EQ(MockTU2_S->getId(), ESIdTU2_S_S1);
      EXPECT_EQ(MockTU2_S->getPatchedIds(), TU2EntityResolutionMapping);

      // Verify TU2's duplicate external entities are NOT in S1
      EXPECT_EQ(S1Data.find(EIdTU2_Q_External), S1Data.end());
    }

    // S2 Tests
    {
      SummaryName S2("S2");
      ASSERT_NE(Data.find(S2), Data.end());

      const auto &S2Data = Data.at(S2);
      // S2 should contain: TU1(Y,Z,B,C,Q,R) + TU2(X,W,A,D,S) = 11 entities
      // Note: TU2's P and Q external entities are dropped because TU1 already
      // has them
      EXPECT_EQ(S2Data.size(), 11u);

      // Verify TU1 entities in S2
      EXPECT_NE(S2Data.find(EIdTU1_Y_None), S2Data.end());
      auto *MockTU1_Y = static_cast<MockEntitySummaryEncoding *>(
          S2Data.at(EIdTU1_Y_None).get());
      EXPECT_EQ(MockTU1_Y->getId(), ESIdTU1_Y_S2);
      EXPECT_EQ(MockTU1_Y->getPatchedIds(), TU1EntityResolutionMapping);

      EXPECT_NE(S2Data.find(EIdTU1_Z_None), S2Data.end());
      auto *MockTU1_Z = static_cast<MockEntitySummaryEncoding *>(
          S2Data.at(EIdTU1_Z_None).get());
      EXPECT_EQ(MockTU1_Z->getId(), ESIdTU1_Z_S2);
      EXPECT_EQ(MockTU1_Z->getPatchedIds(), TU1EntityResolutionMapping);

      EXPECT_NE(S2Data.find(EIdTU1_B_Internal), S2Data.end());
      auto *MockTU1_B = static_cast<MockEntitySummaryEncoding *>(
          S2Data.at(EIdTU1_B_Internal).get());
      EXPECT_EQ(MockTU1_B->getId(), ESIdTU1_B_S2);
      EXPECT_EQ(MockTU1_B->getPatchedIds(), TU1EntityResolutionMapping);

      EXPECT_NE(S2Data.find(EIdTU1_C_Internal), S2Data.end());
      auto *MockTU1_C = static_cast<MockEntitySummaryEncoding *>(
          S2Data.at(EIdTU1_C_Internal).get());
      EXPECT_EQ(MockTU1_C->getId(), ESIdTU1_C_S2);
      EXPECT_EQ(MockTU1_C->getPatchedIds(), TU1EntityResolutionMapping);

      EXPECT_NE(S2Data.find(EIdTU1_Q_External), S2Data.end());
      auto *MockTU1_Q = static_cast<MockEntitySummaryEncoding *>(
          S2Data.at(EIdTU1_Q_External).get());
      EXPECT_EQ(MockTU1_Q->getId(), ESIdTU1_Q_S2);
      EXPECT_EQ(MockTU1_Q->getPatchedIds(), TU1EntityResolutionMapping);

      EXPECT_NE(S2Data.find(EIdTU1_R_External), S2Data.end());
      auto *MockTU1_R = static_cast<MockEntitySummaryEncoding *>(
          S2Data.at(EIdTU1_R_External).get());
      EXPECT_EQ(MockTU1_R->getId(), ESIdTU1_R_S2);
      EXPECT_EQ(MockTU1_R->getPatchedIds(), TU1EntityResolutionMapping);

      // Verify TU2 entities in S2
      EXPECT_NE(S2Data.find(EIdTU2_X_None), S2Data.end());
      auto *MockTU2_X = static_cast<MockEntitySummaryEncoding *>(
          S2Data.at(EIdTU2_X_None).get());
      EXPECT_EQ(MockTU2_X->getId(), ESIdTU2_X_S2);
      EXPECT_EQ(MockTU2_X->getPatchedIds(), TU2EntityResolutionMapping);

      EXPECT_NE(S2Data.find(EIdTU2_W_None), S2Data.end());
      auto *MockTU2_W = static_cast<MockEntitySummaryEncoding *>(
          S2Data.at(EIdTU2_W_None).get());
      EXPECT_EQ(MockTU2_W->getId(), ESIdTU2_W_S2);
      EXPECT_EQ(MockTU2_W->getPatchedIds(), TU2EntityResolutionMapping);

      EXPECT_NE(S2Data.find(EIdTU2_A_Internal), S2Data.end());
      auto *MockTU2_A = static_cast<MockEntitySummaryEncoding *>(
          S2Data.at(EIdTU2_A_Internal).get());
      EXPECT_EQ(MockTU2_A->getId(), ESIdTU2_A_S2);
      EXPECT_EQ(MockTU2_A->getPatchedIds(), TU2EntityResolutionMapping);

      EXPECT_NE(S2Data.find(EIdTU2_D_Internal), S2Data.end());
      auto *MockTU2_D = static_cast<MockEntitySummaryEncoding *>(
          S2Data.at(EIdTU2_D_Internal).get());
      EXPECT_EQ(MockTU2_D->getId(), ESIdTU2_D_S2);
      EXPECT_EQ(MockTU2_D->getPatchedIds(), TU2EntityResolutionMapping);

      EXPECT_NE(S2Data.find(EIdTU2_S_External), S2Data.end());
      auto *MockTU2_S = static_cast<MockEntitySummaryEncoding *>(
          S2Data.at(EIdTU2_S_External).get());
      EXPECT_EQ(MockTU2_S->getId(), ESIdTU2_S_S2);
      EXPECT_EQ(MockTU2_S->getPatchedIds(), TU2EntityResolutionMapping);

      // Verify TU2's duplicate external entities are NOT in S2
      EXPECT_EQ(S2Data.find(EIdTU2_P_External), S2Data.end());
    }
  }
}

} // namespace

} // namespace clang::ssaf
