//===- EntityLinkerTest.cpp -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/ScalableStaticAnalysis/Core/EntityLinker/EntityLinker.h"
#include "TestFixture.h"
#include "clang/ScalableStaticAnalysis/Core/EntityLinker/EntitySummaryEncoding.h"
#include "clang/ScalableStaticAnalysis/Core/EntityLinker/LUSummaryEncoding.h"
#include "clang/ScalableStaticAnalysis/Core/EntityLinker/TUSummaryEncoding.h"
#include "clang/ScalableStaticAnalysis/Core/Model/BuildNamespace.h"
#include "clang/ScalableStaticAnalysis/Core/Model/EntityId.h"
#include "clang/ScalableStaticAnalysis/Core/Model/EntityIdTable.h"
#include "clang/ScalableStaticAnalysis/Core/Model/EntityLinkage.h"
#include "clang/ScalableStaticAnalysis/Core/Model/EntityName.h"
#include "clang/ScalableStaticAnalysis/Core/Model/SummaryName.h"
#include "llvm/Testing/Support/Error.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <memory>

using namespace clang::ssaf;
using namespace llvm;
using ::testing::HasSubstr;
using ::testing::Not;
using ::testing::PrintToString;

namespace {

class MockEntitySummaryEncoding : public EntitySummaryEncoding {
public:
  /// \param Payload Distinguishes encodings for the ODR mismatch check; two
  ///        mocks compare equal iff their payloads match.
  explicit MockEntitySummaryEncoding(int Payload = 0)
      : Id(++Index), Payload(Payload) {}

  size_t getId() const { return Id; }

  llvm::Error patch(const EntityResolutionMap &Resolution) override {
    PatchedIds = Resolution;
    return llvm::Error::success();
  }

  const void *getEncodingKind() const override { return &Kind; }

  bool equals(const EntitySummaryEncoding &Other) const override {
    if (Other.getEncodingKind() != getEncodingKind()) {
      return false;
    }
    return Payload ==
           static_cast<const MockEntitySummaryEncoding &>(Other).Payload;
  }

  const EntityResolutionMap &getPatchedIds() const { return PatchedIds; }

  static size_t Index;

private:
  static const char Kind;

  size_t Id;
  int Payload;
  EntityResolutionMap PatchedIds;
};

size_t MockEntitySummaryEncoding::Index = 0;
const char MockEntitySummaryEncoding::Kind = 0;

class EntityLinkerTest : public TestFixture {
protected:
  static constexpr EntityLinkage NoneLinkage = EntityLinkage(
      EntityLinkageType::None, EntityBinding::Strong, EntityCoalescing::None,
      EntityVisibility::Default, EntityDefinitionKind::Definition);
  static constexpr EntityLinkage InternalLinkage =
      EntityLinkage(EntityLinkageType::Internal, EntityBinding::Strong,
                    EntityCoalescing::None, EntityVisibility::Default,
                    EntityDefinitionKind::Definition);
  static constexpr EntityLinkage ExternalLinkage =
      EntityLinkage(EntityLinkageType::External, EntityBinding::Strong,
                    EntityCoalescing::None, EntityVisibility::Default,
                    EntityDefinitionKind::Definition);

  void SetUp() override {
    // This ensures that the MockEntitySummary id assignment does not
    // accidentally depend on test execution order.
    MockEntitySummaryEncoding::Index = 0;
  }

  std::unique_ptr<TUSummaryEncoding> createTUSummaryEncoding(
      BuildNamespaceKind Kind, llvm::StringRef Name,
      llvm::Triple TargetTriple = llvm::Triple("arm64-apple-macosx")) {
    return std::make_unique<TUSummaryEncoding>(std::move(TargetTriple),
                                               BuildNamespace(Kind, Name));
  }

  size_t addSummaryData(TUSummaryEncoding &TU, EntityId EId,
                        llvm::StringRef SummaryNameStr, int Payload = 0) {
    SummaryName SN(SummaryNameStr.str());
    auto Summary = std::make_unique<MockEntitySummaryEncoding>(Payload);
    const size_t ESId = Summary->getId();
    getData(TU)[SN][EId] = std::move(Summary);
    return ESId;
  }

  EntityId addEntity(TUSummaryEncoding &TU, llvm::StringRef USR,
                     EntityLinkage Linkage) {
    EntityName Name(USR, "", NestedBuildNamespace());
    EntityId Id = getIdTable(TU).getId(Name);
    getLinkageTable(TU).insert({Id, Linkage});
    return Id;
  }
};

// ============================================================================
// Entity ID Table Matchers
// ============================================================================

MATCHER_P(ContainsEntity, EntityName,
          std::string(negation ? "does not contain" : "contains") +
              " entity '" + PrintToString(EntityName) + "'") {
  return arg.contains(EntityName);
}

MATCHER_P(IdTableHasSize, ExpectedCount,
          std::string("has ") + PrintToString(ExpectedCount) + " entities") {
  if (arg.count() != ExpectedCount) {
    *result_listener << "has " << arg.count() << " entities";
    return false;
  }
  return true;
}

// ============================================================================
// Linkage Table Matchers
// ============================================================================

MATCHER_P2(EntityHasLinkage, EId, ExpectedLinkage,
           std::string("entity has ") + PrintToString(ExpectedLinkage) +
               " linkage") {
  auto It = arg.find(EId);
  if (It == arg.end()) {
    *result_listener << "entity " << PrintToString(EId)
                     << " not found in linkage table";
    return false;
  }

  const EntityLinkage &ActualLinkage = It->second;
  if (ActualLinkage != ExpectedLinkage) {
    *result_listener << "entity " << PrintToString(EId) << " has linkage "
                     << PrintToString(ActualLinkage);
    return false;
  }

  return true;
}

MATCHER_P(LinkageTableHasSize, ExpectedSize,
          std::string("linkage table has size ") +
              PrintToString(ExpectedSize)) {
  if (arg.size() != ExpectedSize) {
    *result_listener << "has size " << arg.size();
    return false;
  }
  return true;
}

// ============================================================================
// Summary Data Matchers
// ============================================================================

MATCHER_P3(HasSummaryData, EId, ExpectedMockId, ExpectedResolutionMapping,
           std::string("has summary data for entity with expected mock ID ") +
               PrintToString(ExpectedMockId)) {

  auto It = arg.find(EId);
  if (It == arg.end()) {
    *result_listener << "entity " << PrintToString(EId)
                     << " not found in summary data";
    return false;
  }

  auto *Mock = static_cast<const MockEntitySummaryEncoding *>(It->second.get());

  if (Mock->getId() != ExpectedMockId) {
    *result_listener << "entity " << PrintToString(EId) << " has mock ID "
                     << Mock->getId() << " (expected " << ExpectedMockId << ")";
    return false;
  }

  if (Mock->getPatchedIds() != ExpectedResolutionMapping) {
    *result_listener << "entity " << PrintToString(EId)
                     << " has different resolution mapping";
    return false;
  }

  return true;
}

MATCHER_P(SummaryDataHasSize, ExpectedSize,
          std::string("summary data has size ") + PrintToString(ExpectedSize)) {
  if (arg.size() != ExpectedSize) {
    *result_listener << "has size " << arg.size();
    return false;
  }
  return true;
}

// ============================================================================
// ENTITY LINKER TESTS
// ============================================================================

TEST_F(EntityLinkerTest, CreatesEmptyLinker) {
  NestedBuildNamespace LUNamespace(
      {BuildNamespace(BuildNamespaceKind::LinkUnit, "LU")});

  EntityLinker Linker(llvm::Triple("arm64-apple-macosx"), LUNamespace);

  const auto Output = std::move(Linker).takeOutput();
  EXPECT_EQ(getIdTable(Output).count(), 0u);
  EXPECT_EQ(getLinkageTable(Output).size(), 0u);
  EXPECT_EQ(getData(Output).size(), 0u);
}

TEST_F(EntityLinkerTest, LinksEmptyTranslationUnit) {
  NestedBuildNamespace LUNamespace(
      {BuildNamespace(BuildNamespaceKind::LinkUnit, "LU")});

  EntityLinker Linker(llvm::Triple("arm64-apple-macosx"), LUNamespace);

  auto TUEmpty =
      createTUSummaryEncoding(BuildNamespaceKind::CompilationUnit, "TUEmpty");

  EXPECT_THAT_ERROR(Linker.link(std::move(TUEmpty)), llvm::Succeeded());

  const auto Output = std::move(Linker).takeOutput();
  EXPECT_EQ(getIdTable(Output).count(), 0u);
  EXPECT_EQ(getLinkageTable(Output).size(), 0u);
  EXPECT_EQ(getData(Output).size(), 0u);
}

TEST_F(EntityLinkerTest, LinksOneTranslationUnit) {
  NestedBuildNamespace LUNamespace(
      {BuildNamespace(BuildNamespaceKind::LinkUnit, "LU")});

  EntityLinker Linker(llvm::Triple("arm64-apple-macosx"), LUNamespace);

  auto TU = createTUSummaryEncoding(BuildNamespaceKind::CompilationUnit, "TU");

  const auto TU_A_Id = addEntity(*TU, "A", NoneLinkage);
  const auto TU_A_S1_Data = addSummaryData(*TU, TU_A_Id, "S1");
  const auto TU_A_S2_Data = addSummaryData(*TU, TU_A_Id, "S2");

  const auto TU_B_Id = addEntity(*TU, "B", InternalLinkage);
  const auto TU_B_S1_Data = addSummaryData(*TU, TU_B_Id, "S1");
  const auto TU_B_S2_Data = addSummaryData(*TU, TU_B_Id, "S2");

  const auto TU_C_Id = addEntity(*TU, "C", ExternalLinkage);
  const auto TU_C_S1_Data = addSummaryData(*TU, TU_C_Id, "S1");

  const auto TU_D_Id = addEntity(*TU, "D", ExternalLinkage);
  const auto TU_D_S2_Data = addSummaryData(*TU, TU_D_Id, "S2");

  const BuildNamespace TUNamespace = getTUNamespace(*TU);

  ASSERT_THAT_ERROR(Linker.link(std::move(TU)), llvm::Succeeded());

  const auto Output = std::move(Linker).takeOutput();
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
    ASSERT_THAT(LinkageTable, EntityHasLinkage(LU_A_Id, NoneLinkage));
    ASSERT_THAT(LinkageTable, EntityHasLinkage(LU_B_Id, InternalLinkage));
    ASSERT_THAT(LinkageTable, EntityHasLinkage(LU_C_Id, ExternalLinkage));
    ASSERT_THAT(LinkageTable, EntityHasLinkage(LU_D_Id, ExternalLinkage));
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
      ASSERT_THAT(S1Data, HasSummaryData(LU_C_Id, TU_C_S1_Data, Resolution));
    }

    // S2 Data Tests.
    {
      SummaryName S2("S2");
      ASSERT_NE(Data.find(S2), Data.end());
      const auto &S2Data = Data.at(S2);

      ASSERT_THAT(S2Data, SummaryDataHasSize(3u));
      ASSERT_THAT(S2Data, HasSummaryData(LU_A_Id, TU_A_S2_Data, Resolution));
      ASSERT_THAT(S2Data, HasSummaryData(LU_B_Id, TU_B_S2_Data, Resolution));
      ASSERT_THAT(S2Data, HasSummaryData(LU_D_Id, TU_D_S2_Data, Resolution));
    }
  }
}

TEST_F(EntityLinkerTest, LinksTwoTranslationUnits) {
  NestedBuildNamespace LUNamespace(
      {BuildNamespace(BuildNamespaceKind::LinkUnit, "LU")});

  // The two TUs below both define the external entities P, Q and R, so the
  // linker is asked to warn about the multiple definitions rather than reject
  // them, leaving the duplicate summary data to be dropped as usual.
  EntityLinker Linker(llvm::Triple("arm64-apple-macosx"), LUNamespace,
                      /*WarnOnMultipleDefinitions=*/true);

  auto TU1 =
      createTUSummaryEncoding(BuildNamespaceKind::CompilationUnit, "TU1");

  // None linkage entities in TU1
  const auto TU1_X_Id = addEntity(*TU1, "X", NoneLinkage);
  const auto TU1_X_S1_Data = addSummaryData(*TU1, TU1_X_Id, "S1");

  const auto TU1_Y_Id = addEntity(*TU1, "Y", NoneLinkage);
  const auto TU1_Y_S2_Data = addSummaryData(*TU1, TU1_Y_Id, "S2");

  const auto TU1_Z_Id = addEntity(*TU1, "Z", NoneLinkage);
  const auto TU1_Z_S1_Data = addSummaryData(*TU1, TU1_Z_Id, "S1");
  const auto TU1_Z_S2_Data = addSummaryData(*TU1, TU1_Z_Id, "S2");

  // Internal linkage entities in TU1
  const auto TU1_A_Id = addEntity(*TU1, "A", InternalLinkage);
  const auto TU1_A_S1_Data = addSummaryData(*TU1, TU1_A_Id, "S1");

  const auto TU1_B_Id = addEntity(*TU1, "B", InternalLinkage);
  const auto TU1_B_S2_Data = addSummaryData(*TU1, TU1_B_Id, "S2");

  const auto TU1_C_Id = addEntity(*TU1, "C", InternalLinkage);
  const auto TU1_C_S1_Data = addSummaryData(*TU1, TU1_C_Id, "S1");
  const auto TU1_C_S2_Data = addSummaryData(*TU1, TU1_C_Id, "S2");

  // External linkage entities in TU1
  const auto TU1_P_Id = addEntity(*TU1, "P", ExternalLinkage);
  const auto TU1_P_S1_Data = addSummaryData(*TU1, TU1_P_Id, "S1");

  const auto TU1_Q_Id = addEntity(*TU1, "Q", ExternalLinkage);
  const auto TU1_Q_S2_Data = addSummaryData(*TU1, TU1_Q_Id, "S2");

  const auto TU1_R_Id = addEntity(*TU1, "R", ExternalLinkage);
  const auto TU1_R_S1_Data = addSummaryData(*TU1, TU1_R_Id, "S1");
  const auto TU1_R_S2_Data = addSummaryData(*TU1, TU1_R_Id, "S2");

  const BuildNamespace TU1Namespace = getTUNamespace(*TU1);

  ASSERT_THAT_ERROR(Linker.link(std::move(TU1)), llvm::Succeeded());

  auto TU2 =
      createTUSummaryEncoding(BuildNamespaceKind::CompilationUnit, "TU2");

  // None linkage entities in TU2 - includes duplicates and uniques
  const auto TU2_X_Id = addEntity(*TU2, "X", NoneLinkage);
  const auto TU2_X_S2_Data = addSummaryData(*TU2, TU2_X_Id, "S2");

  const auto TU2_Y_Id = addEntity(*TU2, "Y", NoneLinkage);
  const auto TU2_Y_S1_Data = addSummaryData(*TU2, TU2_Y_Id, "S1");

  const auto TU2_W_Id = addEntity(*TU2, "W", NoneLinkage);
  const auto TU2_W_S1_Data = addSummaryData(*TU2, TU2_W_Id, "S1");
  const auto TU2_W_S2_Data = addSummaryData(*TU2, TU2_W_Id, "S2");

  // Internal linkage entities in TU2 - includes duplicates and unique
  const auto TU2_A_Id = addEntity(*TU2, "A", InternalLinkage);
  const auto TU2_A_S2_Data = addSummaryData(*TU2, TU2_A_Id, "S2");

  const auto TU2_B_Id = addEntity(*TU2, "B", InternalLinkage);
  const auto TU2_B_S1_Data = addSummaryData(*TU2, TU2_B_Id, "S1");

  const auto TU2_D_Id = addEntity(*TU2, "D", InternalLinkage);
  const auto TU2_D_S1_Data = addSummaryData(*TU2, TU2_D_Id, "S1");
  const auto TU2_D_S2_Data = addSummaryData(*TU2, TU2_D_Id, "S2");

  // External linkage entities in TU2 - includes duplicates (will be dropped)
  // and uniques
  const auto TU2_P_Id = addEntity(*TU2, "P", ExternalLinkage);
  const auto TU2_P_S2_Data = addSummaryData(*TU2, TU2_P_Id, "S2");

  const auto TU2_Q_Id = addEntity(*TU2, "Q", ExternalLinkage);
  const auto TU2_Q_S1_Data = addSummaryData(*TU2, TU2_Q_Id, "S1");

  const auto TU2_R_Id = addEntity(*TU2, "R", ExternalLinkage);
  const auto TU2_R_S1_Data = addSummaryData(*TU2, TU2_R_Id, "S1");
  const auto TU2_R_S2_Data = addSummaryData(*TU2, TU2_R_Id, "S2");

  const auto TU2_S_Id = addEntity(*TU2, "S", ExternalLinkage);
  const auto TU2_S_S1_Data = addSummaryData(*TU2, TU2_S_Id, "S1");
  const auto TU2_S_S2_Data = addSummaryData(*TU2, TU2_S_Id, "S2");

  const BuildNamespace TU2Namespace = getTUNamespace(*TU2);

  ASSERT_THAT_ERROR(Linker.link(std::move(TU2)), llvm::Succeeded());

  const auto Output = std::move(Linker).takeOutput();
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

    ASSERT_THAT(LinkageTable, EntityHasLinkage(LU_TU1_X_Id, NoneLinkage));
    ASSERT_THAT(LinkageTable, EntityHasLinkage(LU_TU1_Y_Id, NoneLinkage));
    ASSERT_THAT(LinkageTable, EntityHasLinkage(LU_TU1_Z_Id, NoneLinkage));
    ASSERT_THAT(LinkageTable, EntityHasLinkage(LU_TU2_X_Id, NoneLinkage));
    ASSERT_THAT(LinkageTable, EntityHasLinkage(LU_TU2_Y_Id, NoneLinkage));
    ASSERT_THAT(LinkageTable, EntityHasLinkage(LU_TU2_W_Id, NoneLinkage));

    ASSERT_THAT(LinkageTable, EntityHasLinkage(LU_TU1_A_Id, InternalLinkage));
    ASSERT_THAT(LinkageTable, EntityHasLinkage(LU_TU1_B_Id, InternalLinkage));
    ASSERT_THAT(LinkageTable, EntityHasLinkage(LU_TU1_C_Id, InternalLinkage));
    ASSERT_THAT(LinkageTable, EntityHasLinkage(LU_TU2_A_Id, InternalLinkage));
    ASSERT_THAT(LinkageTable, EntityHasLinkage(LU_TU2_B_Id, InternalLinkage));
    ASSERT_THAT(LinkageTable, EntityHasLinkage(LU_TU2_D_Id, InternalLinkage));

    ASSERT_THAT(LinkageTable, EntityHasLinkage(LU_P_Id, ExternalLinkage));
    ASSERT_THAT(LinkageTable, EntityHasLinkage(LU_Q_Id, ExternalLinkage));
    ASSERT_THAT(LinkageTable, EntityHasLinkage(LU_R_Id, ExternalLinkage));
    ASSERT_THAT(LinkageTable, EntityHasLinkage(LU_S_Id, ExternalLinkage));
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
        {TU2_R_Id, LU_R_Id},     {TU2_S_Id, LU_S_Id}};

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
                  Not(HasSummaryData(LU_R_Id, TU2_R_S1_Data, TU2Resolution)));
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
                  Not(HasSummaryData(LU_R_Id, TU2_R_S2_Data, TU2Resolution)));
      ASSERT_THAT(S2Data,
                  HasSummaryData(LU_S_Id, TU2_S_S2_Data, TU2Resolution));
    }
  }
}

TEST_F(EntityLinkerTest, RejectsDuplicateTUSummary) {
  NestedBuildNamespace LUNamespace(
      {BuildNamespace(BuildNamespaceKind::LinkUnit, "LU")});

  EntityLinker Linker(llvm::Triple("arm64-apple-macosx"), LUNamespace);

  auto TU1 = createTUSummaryEncoding(BuildNamespaceKind::CompilationUnit, "TU");

  ASSERT_THAT_ERROR(Linker.link(std::move(TU1)), llvm::Succeeded());

  auto TU2 = createTUSummaryEncoding(BuildNamespaceKind::CompilationUnit, "TU");

  ASSERT_THAT_ERROR(Linker.link(std::move(TU2)),
                    llvm::FailedWithMessage(
                        HasSubstr("failed to link TU summary: duplicate "
                                  "BuildNamespace(CompilationUnit, TU)")));
}

// Reproduces a crash when linking internal-linkage entities
// (e.g. "static inline" functions) that share the same USR across TUs.
TEST_F(EntityLinkerTest, InternalLinkageWithEmptyNamespaceAcrossTUs) {
  constexpr auto LinkUnit = BuildNamespaceKind::LinkUnit;
  constexpr auto CompilationUnit = BuildNamespaceKind::CompilationUnit;
  EntityLinker Linker(llvm::Triple("arm64-apple-macosx"),
                      NestedBuildNamespace{BuildNamespace(LinkUnit, "LU")});

  auto TU1 = createTUSummaryEncoding(CompilationUnit, "TU1");
  addEntity(*TU1, "some_static_inline", InternalLinkage);
  ASSERT_THAT_ERROR(Linker.link(std::move(TU1)), llvm::Succeeded());

  auto TU2 = createTUSummaryEncoding(CompilationUnit, "TU2");
  addEntity(*TU2, "some_static_inline", InternalLinkage);
  ASSERT_THAT_ERROR(Linker.link(std::move(TU2)), llvm::Succeeded());

  // Check that the two internal symbols are not merged.
  const auto Output = std::move(Linker).takeOutput();
  const auto &IdTable = getIdTable(Output);

  NestedBuildNamespace TU1NS{{{CompilationUnit, "TU1"}, {LinkUnit, "LU"}}};
  NestedBuildNamespace TU2NS{{{CompilationUnit, "TU2"}, {LinkUnit, "LU"}}};
  EntityName ExpectedTU1Name("some_static_inline", "", TU1NS);
  EntityName ExpectedTU2Name("some_static_inline", "", TU2NS);

  ASSERT_EQ(IdTable.count(), 2u);
  EXPECT_THAT(IdTable, ContainsEntity(ExpectedTU1Name));
  EXPECT_THAT(IdTable, ContainsEntity(ExpectedTU2Name));
}

// ============================================================================
// MULTIPLE DEFINITION TESTS
// ============================================================================

TEST_F(EntityLinkerTest, FailsOnMultipleDefinitions) {
  constexpr auto LinkUnit = BuildNamespaceKind::LinkUnit;
  constexpr auto CompilationUnit = BuildNamespaceKind::CompilationUnit;

  EXPECT_DEATH(
      {
        EntityLinker Linker(
            llvm::Triple("arm64-apple-macosx"),
            NestedBuildNamespace{BuildNamespace(LinkUnit, "LU")});

        auto TU1 = createTUSummaryEncoding(CompilationUnit, "TU1");
        addEntity(*TU1, "P", ExternalLinkage);
        cantFail(Linker.link(std::move(TU1)));

        auto TU2 = createTUSummaryEncoding(CompilationUnit, "TU2");
        addEntity(*TU2, "P", ExternalLinkage);
        cantFail(Linker.link(std::move(TU2)));
      },
      "multiple definition of");
}

TEST_F(EntityLinkerTest, WarnsOnMultipleDefinitions) {
  constexpr auto LinkUnit = BuildNamespaceKind::LinkUnit;
  constexpr auto CompilationUnit = BuildNamespaceKind::CompilationUnit;

  EntityLinker Linker(llvm::Triple("arm64-apple-macosx"),
                      NestedBuildNamespace{BuildNamespace(LinkUnit, "LU")},
                      /*WarnOnMultipleDefinitions=*/true);

  auto TU1 = createTUSummaryEncoding(CompilationUnit, "TU1");
  const auto TU1_P_Id = addEntity(*TU1, "P", ExternalLinkage);
  const auto TU1_P_S1_Data = addSummaryData(*TU1, TU1_P_Id, "S1");
  ASSERT_THAT_ERROR(Linker.link(std::move(TU1)), llvm::Succeeded());

  auto TU2 = createTUSummaryEncoding(CompilationUnit, "TU2");
  const auto TU2_P_Id = addEntity(*TU2, "P", ExternalLinkage);
  addSummaryData(*TU2, TU2_P_Id, "S1");

  testing::internal::CaptureStderr();
  ASSERT_THAT_ERROR(Linker.link(std::move(TU2)), llvm::Succeeded());
  const std::string Stderr = testing::internal::GetCapturedStderr();

  EXPECT_THAT(Stderr, HasSubstr("warning: multiple definition of "
                                "EntityName(P, , NestedBuildNamespace(["
                                "BuildNamespace(LinkUnit, LU)]))"));
  EXPECT_THAT(Stderr, HasSubstr("in NestedBuildNamespace(["
                                "BuildNamespace(CompilationUnit, TU2)])"));

  // The extra definition is reported but still dropped: TU1 defined P first,
  // so its summary data is the data that survives.
  const auto Output = std::move(Linker).takeOutput();
  const auto &IdTable = getIdTable(Output);

  NestedBuildNamespace LUNamespace{{{LinkUnit, "LU"}}};
  EntityName LU_P_Name("P", "", LUNamespace);
  ASSERT_THAT(IdTable, ContainsEntity(LU_P_Name));

  const auto LU_P_Id = getEntities(IdTable).at(LU_P_Name);
  const std::map<EntityId, EntityId> TU1Resolution = {{TU1_P_Id, LU_P_Id}};
  EXPECT_THAT(getData(Output).at(SummaryName("S1")),
              HasSummaryData(LU_P_Id, TU1_P_S1_Data, TU1Resolution));
}

TEST_F(EntityLinkerTest, AcceptsDeclarationAlongsideDefinition) {
  constexpr auto LinkUnit = BuildNamespaceKind::LinkUnit;
  constexpr auto CompilationUnit = BuildNamespaceKind::CompilationUnit;
  constexpr EntityLinkage ExternalDeclaration =
      EntityLinkage(EntityLinkageType::External, EntityBinding::Undefined,
                    EntityCoalescing::None, EntityVisibility::Default,
                    EntityDefinitionKind::Declaration);

  EntityLinker Linker(llvm::Triple("arm64-apple-macosx"),
                      NestedBuildNamespace{BuildNamespace(LinkUnit, "LU")});

  auto TU1 = createTUSummaryEncoding(CompilationUnit, "TU1");
  addEntity(*TU1, "P", ExternalLinkage);
  ASSERT_THAT_ERROR(Linker.link(std::move(TU1)), llvm::Succeeded());

  // A TU that only references P must not be mistaken for a second definition.
  auto TU2 = createTUSummaryEncoding(CompilationUnit, "TU2");
  addEntity(*TU2, "P", ExternalDeclaration);
  ASSERT_THAT_ERROR(Linker.link(std::move(TU2)), llvm::Succeeded());

  const auto Output = std::move(Linker).takeOutput();
  NestedBuildNamespace LUNamespace{{{LinkUnit, "LU"}}};
  EntityName LU_P_Name("P", "", LUNamespace);

  const auto &IdTable = getIdTable(Output);
  ASSERT_THAT(IdTable, IdTableHasSize(1u));
  ASSERT_THAT(IdTable, ContainsEntity(LU_P_Name));

  // The definition wins the reconciliation.
  EXPECT_THAT(
      getLinkageTable(Output),
      EntityHasLinkage(getEntities(IdTable).at(LU_P_Name), ExternalLinkage));
}

// ============================================================================
// TARGET VALIDATION TESTS
// ============================================================================

TEST_F(EntityLinkerTest, RejectsTUTargetingADifferentPlatform) {
  constexpr auto LinkUnit = BuildNamespaceKind::LinkUnit;
  constexpr auto CompilationUnit = BuildNamespaceKind::CompilationUnit;

  EntityLinker Linker(llvm::Triple("arm64-apple-macosx"),
                      NestedBuildNamespace{BuildNamespace(LinkUnit, "LU")});

  auto TU = createTUSummaryEncoding(CompilationUnit, "TU",
                                    llvm::Triple("x86_64-unknown-linux-gnu"));

  // Applying Mach-O rules to an ELF summary is silently wrong, since the two
  // disagree about whether a weak definition outranks a common one.
  EXPECT_THAT_ERROR(
      Linker.link(std::move(TU)),
      llvm::FailedWithMessage(AllOf(
          HasSubstr("targets 'x86_64-unknown-linux-gnu', which resolves "
                    "symbols as ELF"),
          HasSubstr("link unit targets 'arm64-apple-macosx', which resolves "
                    "them as Mach-O"))));
}

TEST_F(EntityLinkerTest, AcceptsTUDifferingOnlyInOSVersion) {
  constexpr auto LinkUnit = BuildNamespaceKind::LinkUnit;
  constexpr auto CompilationUnit = BuildNamespaceKind::CompilationUnit;

  // The OS version does not affect symbol resolution, so it is not compared.
  EntityLinker Linker(llvm::Triple("arm64-apple-macosx14.0"),
                      NestedBuildNamespace{BuildNamespace(LinkUnit, "LU")});

  auto TU = createTUSummaryEncoding(CompilationUnit, "TU",
                                    llvm::Triple("arm64-apple-macosx15.0"));
  EXPECT_THAT_ERROR(Linker.link(std::move(TU)), llvm::Succeeded());
}

TEST_F(EntityLinkerTest, RejectsProtectedVisibilityOnMachO) {
  constexpr auto LinkUnit = BuildNamespaceKind::LinkUnit;
  constexpr auto CompilationUnit = BuildNamespaceKind::CompilationUnit;
  constexpr EntityLinkage ProtectedDef =
      EntityLinkage(EntityLinkageType::External, EntityBinding::Strong,
                    EntityCoalescing::None, EntityVisibility::Protected,
                    EntityDefinitionKind::Definition);

  // clang warns and downgrades protected visibility on a Mach-O target, so a
  // summary carrying it did not come from the compiler.
  EXPECT_DEATH(
      {
        EntityLinker Linker(
            llvm::Triple("arm64-apple-macosx"),
            NestedBuildNamespace{BuildNamespace(LinkUnit, "LU")});
        auto TU = createTUSummaryEncoding(CompilationUnit, "TU");
        addEntity(*TU, "P", ProtectedDef);
        cantFail(Linker.link(std::move(TU)));
      },
      "which a Mach-O target cannot represent");
}

TEST_F(EntityLinkerTest, CoercesHiddenVisibilityOnCOFF) {
  constexpr auto LinkUnit = BuildNamespaceKind::LinkUnit;
  constexpr auto CompilationUnit = BuildNamespaceKind::CompilationUnit;
  constexpr EntityLinkage HiddenDef =
      EntityLinkage(EntityLinkageType::External, EntityBinding::Strong,
                    EntityCoalescing::None, EntityVisibility::Hidden,
                    EntityDefinitionKind::Definition);

  // COFF drops visibility at emission and clang accepts the attribute
  // silently, so portable source legitimately produces this. Coerce rather
  // than reject.
  EntityLinker Linker(llvm::Triple("x86_64-pc-windows-msvc"),
                      NestedBuildNamespace{BuildNamespace(LinkUnit, "LU")});

  auto TU = createTUSummaryEncoding(CompilationUnit, "TU",
                                    llvm::Triple("x86_64-pc-windows-msvc"));
  addEntity(*TU, "P", HiddenDef);
  ASSERT_THAT_ERROR(Linker.link(std::move(TU)), llvm::Succeeded());

  const auto Output = std::move(Linker).takeOutput();
  NestedBuildNamespace LUNamespace{{{LinkUnit, "LU"}}};
  EntityName LU_P_Name("P", "", LUNamespace);
  const auto &IdTable = getIdTable(Output);
  ASSERT_THAT(IdTable, ContainsEntity(LU_P_Name));

  constexpr EntityLinkage CoercedDef =
      EntityLinkage(EntityLinkageType::External, EntityBinding::Strong,
                    EntityCoalescing::None, EntityVisibility::Default,
                    EntityDefinitionKind::Definition);
  EXPECT_THAT(getLinkageTable(Output),
              EntityHasLinkage(getEntities(IdTable).at(LU_P_Name), CoercedDef));
}

// ============================================================================
// FINALIZATION TESTS
// ============================================================================

TEST_F(EntityLinkerTest, DemotesHiddenEntitiesToInternal) {
  constexpr auto LinkUnit = BuildNamespaceKind::LinkUnit;
  constexpr auto CompilationUnit = BuildNamespaceKind::CompilationUnit;
  constexpr EntityLinkage HiddenDef =
      EntityLinkage(EntityLinkageType::External, EntityBinding::Strong,
                    EntityCoalescing::None, EntityVisibility::Hidden,
                    EntityDefinitionKind::Definition);

  EntityLinker Linker(llvm::Triple("x86_64-unknown-linux-gnu"),
                      NestedBuildNamespace{BuildNamespace(LinkUnit, "LU")});

  auto TU = createTUSummaryEncoding(CompilationUnit, "TU",
                                    llvm::Triple("x86_64-unknown-linux-gnu"));
  addEntity(*TU, "hidden_fn", HiddenDef);
  addEntity(*TU, "exported_fn", ExternalLinkage);
  ASSERT_THAT_ERROR(Linker.link(std::move(TU)), llvm::Succeeded());

  const auto Output = std::move(Linker).takeOutput();
  NestedBuildNamespace LUNamespace{{{LinkUnit, "LU"}}};
  const auto &IdTable = getIdTable(Output);
  const auto &Entities = getEntities(IdTable);

  // A hidden entity is not resolvable past the link unit boundary, which is
  // Internal linkage in our model.
  constexpr EntityLinkage DemotedDef =
      EntityLinkage(EntityLinkageType::Internal, EntityBinding::Strong,
                    EntityCoalescing::None, EntityVisibility::Hidden,
                    EntityDefinitionKind::Definition);
  EXPECT_THAT(
      getLinkageTable(Output),
      EntityHasLinkage(Entities.at(EntityName("hidden_fn", "", LUNamespace)),
                       DemotedDef));

  // The default-visibility entity is untouched.
  EXPECT_THAT(
      getLinkageTable(Output),
      EntityHasLinkage(Entities.at(EntityName("exported_fn", "", LUNamespace)),
                       ExternalLinkage));
}

TEST_F(EntityLinkerTest, KeepsHiddenEntityNamesWhenDemoting) {
  constexpr auto LinkUnit = BuildNamespaceKind::LinkUnit;
  constexpr auto CompilationUnit = BuildNamespaceKind::CompilationUnit;
  constexpr EntityLinkage HiddenDef =
      EntityLinkage(EntityLinkageType::External, EntityBinding::Strong,
                    EntityCoalescing::None, EntityVisibility::Hidden,
                    EntityDefinitionKind::Definition);

  EntityLinker Linker(llvm::Triple("x86_64-unknown-linux-gnu"),
                      NestedBuildNamespace{BuildNamespace(LinkUnit, "LU")});

  auto TU = createTUSummaryEncoding(CompilationUnit, "TU",
                                    llvm::Triple("x86_64-unknown-linux-gnu"));
  addEntity(*TU, "hidden_fn", HiddenDef);
  ASSERT_THAT_ERROR(Linker.link(std::move(TU)), llvm::Succeeded());

  // Demotion changes the linkage type only. Renaming would invalidate every
  // EntityId already patched into the summary data, and is unnecessary because
  // external names are already LU-qualified.
  const auto Output = std::move(Linker).takeOutput();
  NestedBuildNamespace LUNamespace{{{LinkUnit, "LU"}}};
  EXPECT_THAT(getIdTable(Output),
              ContainsEntity(EntityName("hidden_fn", "", LUNamespace)));
}

TEST_F(EntityLinkerTest, IgnoresUnresolvedEntitiesByDefault) {
  constexpr auto LinkUnit = BuildNamespaceKind::LinkUnit;
  constexpr auto CompilationUnit = BuildNamespaceKind::CompilationUnit;
  constexpr EntityLinkage ExternalDeclaration =
      EntityLinkage(EntityLinkageType::External, EntityBinding::Undefined,
                    EntityCoalescing::None, EntityVisibility::Default,
                    EntityDefinitionKind::Declaration);

  // A link unit is usually intermediate, so an unresolved reference is not by
  // itself an error.
  EntityLinker Linker(llvm::Triple("x86_64-unknown-linux-gnu"),
                      NestedBuildNamespace{BuildNamespace(LinkUnit, "LU")});

  auto TU = createTUSummaryEncoding(CompilationUnit, "TU",
                                    llvm::Triple("x86_64-unknown-linux-gnu"));
  addEntity(*TU, "missing", ExternalDeclaration);
  ASSERT_THAT_ERROR(Linker.link(std::move(TU)), llvm::Succeeded());

  testing::internal::CaptureStderr();
  const auto Output = std::move(Linker).takeOutput();
  EXPECT_THAT(testing::internal::GetCapturedStderr(),
              Not(HasSubstr("undefined symbol")));
}

TEST_F(EntityLinkerTest, WarnsOnUnresolvedEntities) {
  constexpr auto LinkUnit = BuildNamespaceKind::LinkUnit;
  constexpr auto CompilationUnit = BuildNamespaceKind::CompilationUnit;
  constexpr EntityLinkage ExternalDeclaration =
      EntityLinkage(EntityLinkageType::External, EntityBinding::Undefined,
                    EntityCoalescing::None, EntityVisibility::Default,
                    EntityDefinitionKind::Declaration);
  constexpr EntityLinkage WeakDeclaration = EntityLinkage(
      EntityLinkageType::External, EntityBinding::Weak, EntityCoalescing::None,
      EntityVisibility::Default, EntityDefinitionKind::Declaration);

  EntityLinker Linker(llvm::Triple("x86_64-unknown-linux-gnu"),
                      NestedBuildNamespace{BuildNamespace(LinkUnit, "LU")},
                      /*WarnOnMultipleDefinitions=*/false,
                      UnresolvedPolicy::Warn);

  auto TU = createTUSummaryEncoding(CompilationUnit, "TU",
                                    llvm::Triple("x86_64-unknown-linux-gnu"));
  addEntity(*TU, "missing", ExternalDeclaration);
  addEntity(*TU, "maybe_missing", WeakDeclaration);
  addEntity(*TU, "defined", ExternalLinkage);
  ASSERT_THAT_ERROR(Linker.link(std::move(TU)), llvm::Succeeded());

  testing::internal::CaptureStderr();
  const auto Output = std::move(Linker).takeOutput();
  const std::string Stderr = testing::internal::GetCapturedStderr();

  EXPECT_THAT(Stderr, HasSubstr("undefined symbol: EntityName(missing"));
  // An undefined weak reference resolves to nothing by design (ELF probe U2).
  EXPECT_THAT(Stderr, Not(HasSubstr("maybe_missing")));
  EXPECT_THAT(Stderr, Not(HasSubstr("EntityName(defined")));
}

TEST_F(EntityLinkerTest, ReportsUnresolvedEntitiesOnlyAfterEveryTUIsLinked) {
  constexpr auto LinkUnit = BuildNamespaceKind::LinkUnit;
  constexpr auto CompilationUnit = BuildNamespaceKind::CompilationUnit;
  constexpr EntityLinkage ExternalDeclaration =
      EntityLinkage(EntityLinkageType::External, EntityBinding::Undefined,
                    EntityCoalescing::None, EntityVisibility::Default,
                    EntityDefinitionKind::Declaration);

  EntityLinker Linker(llvm::Triple("x86_64-unknown-linux-gnu"),
                      NestedBuildNamespace{BuildNamespace(LinkUnit, "LU")},
                      /*WarnOnMultipleDefinitions=*/false,
                      UnresolvedPolicy::Warn);

  // The first TU only declares P; the second defines it. Reporting per-TU
  // would produce a spurious diagnostic here, which is why the check runs at
  // finalization.
  auto TU1 = createTUSummaryEncoding(CompilationUnit, "TU1",
                                     llvm::Triple("x86_64-unknown-linux-gnu"));
  addEntity(*TU1, "P", ExternalDeclaration);
  ASSERT_THAT_ERROR(Linker.link(std::move(TU1)), llvm::Succeeded());

  auto TU2 = createTUSummaryEncoding(CompilationUnit, "TU2",
                                     llvm::Triple("x86_64-unknown-linux-gnu"));
  addEntity(*TU2, "P", ExternalLinkage);
  ASSERT_THAT_ERROR(Linker.link(std::move(TU2)), llvm::Succeeded());

  testing::internal::CaptureStderr();
  const auto Output = std::move(Linker).takeOutput();
  EXPECT_THAT(testing::internal::GetCapturedStderr(),
              Not(HasSubstr("undefined symbol")));
}

TEST_F(EntityLinkerTest, FailsOnUnresolvedEntitiesUnderErrorPolicy) {
  constexpr auto LinkUnit = BuildNamespaceKind::LinkUnit;
  constexpr auto CompilationUnit = BuildNamespaceKind::CompilationUnit;
  constexpr EntityLinkage ExternalDeclaration =
      EntityLinkage(EntityLinkageType::External, EntityBinding::Undefined,
                    EntityCoalescing::None, EntityVisibility::Default,
                    EntityDefinitionKind::Declaration);

  EXPECT_DEATH(
      {
        EntityLinker Linker(
            llvm::Triple("x86_64-unknown-linux-gnu"),
            NestedBuildNamespace{BuildNamespace(LinkUnit, "LU")},
            /*WarnOnMultipleDefinitions=*/false, UnresolvedPolicy::Error);
        auto TU = createTUSummaryEncoding(
            CompilationUnit, "TU", llvm::Triple("x86_64-unknown-linux-gnu"));
        addEntity(*TU, "missing", ExternalDeclaration);
        cantFail(Linker.link(std::move(TU)));
        auto Output = std::move(Linker).takeOutput();
      },
      "undefined symbol");
}

// ============================================================================
// ODR MISMATCH TESTS
// ============================================================================

namespace {
/// An inline definition: strong at the source level, with the ODR guarantee.
constexpr EntityLinkage ODRDefinition = EntityLinkage(
    EntityLinkageType::External, EntityBinding::Strong, EntityCoalescing::ODR,
    EntityVisibility::Default, EntityDefinitionKind::Definition);
} // namespace

TEST_F(EntityLinkerTest, IgnoresODRMismatchByDefault) {
  constexpr auto LinkUnit = BuildNamespaceKind::LinkUnit;
  constexpr auto CompilationUnit = BuildNamespaceKind::CompilationUnit;

  EntityLinker Linker(llvm::Triple("arm64-apple-macosx"),
                      NestedBuildNamespace{BuildNamespace(LinkUnit, "LU")});

  auto TU1 = createTUSummaryEncoding(CompilationUnit, "TU1");
  addSummaryData(*TU1, addEntity(*TU1, "P", ODRDefinition), "S1",
                 /*Payload=*/1);
  ASSERT_THAT_ERROR(Linker.link(std::move(TU1)), llvm::Succeeded());

  auto TU2 = createTUSummaryEncoding(CompilationUnit, "TU2");
  addSummaryData(*TU2, addEntity(*TU2, "P", ODRDefinition), "S1",
                 /*Payload=*/2);

  testing::internal::CaptureStderr();
  ASSERT_THAT_ERROR(Linker.link(std::move(TU2)), llvm::Succeeded());
  EXPECT_THAT(testing::internal::GetCapturedStderr(),
              Not(HasSubstr("differs between")));
}

TEST_F(EntityLinkerTest, WarnsOnODRMismatch) {
  constexpr auto LinkUnit = BuildNamespaceKind::LinkUnit;
  constexpr auto CompilationUnit = BuildNamespaceKind::CompilationUnit;

  EntityLinker Linker(llvm::Triple("arm64-apple-macosx"),
                      NestedBuildNamespace{BuildNamespace(LinkUnit, "LU")},
                      /*WarnOnMultipleDefinitions=*/false,
                      UnresolvedPolicy::Ignore, ODRMismatchPolicy::Warn);

  auto TU1 = createTUSummaryEncoding(CompilationUnit, "TU1");
  addSummaryData(*TU1, addEntity(*TU1, "P", ODRDefinition), "S1",
                 /*Payload=*/1);
  ASSERT_THAT_ERROR(Linker.link(std::move(TU1)), llvm::Succeeded());

  // The same inline function summarized differently in another TU means the
  // two definitions were not in fact identical.
  auto TU2 = createTUSummaryEncoding(CompilationUnit, "TU2");
  addSummaryData(*TU2, addEntity(*TU2, "P", ODRDefinition), "S1",
                 /*Payload=*/2);

  testing::internal::CaptureStderr();
  ASSERT_THAT_ERROR(Linker.link(std::move(TU2)), llvm::Succeeded());
  const std::string Stderr = testing::internal::GetCapturedStderr();

  EXPECT_THAT(Stderr, HasSubstr("SummaryName(S1) summary data for "
                                "EntityName(P, "));
  EXPECT_THAT(Stderr, HasSubstr("definitions are required to be identical"));
}

TEST_F(EntityLinkerTest, AcceptsMatchingODRDefinitions) {
  constexpr auto LinkUnit = BuildNamespaceKind::LinkUnit;
  constexpr auto CompilationUnit = BuildNamespaceKind::CompilationUnit;

  EntityLinker Linker(llvm::Triple("arm64-apple-macosx"),
                      NestedBuildNamespace{BuildNamespace(LinkUnit, "LU")},
                      /*WarnOnMultipleDefinitions=*/false,
                      UnresolvedPolicy::Ignore, ODRMismatchPolicy::Warn);

  auto TU1 = createTUSummaryEncoding(CompilationUnit, "TU1");
  addSummaryData(*TU1, addEntity(*TU1, "P", ODRDefinition), "S1",
                 /*Payload=*/7);
  ASSERT_THAT_ERROR(Linker.link(std::move(TU1)), llvm::Succeeded());

  auto TU2 = createTUSummaryEncoding(CompilationUnit, "TU2");
  addSummaryData(*TU2, addEntity(*TU2, "P", ODRDefinition), "S1",
                 /*Payload=*/7);

  testing::internal::CaptureStderr();
  ASSERT_THAT_ERROR(Linker.link(std::move(TU2)), llvm::Succeeded());
  EXPECT_THAT(testing::internal::GetCapturedStderr(),
              Not(HasSubstr("differs between")));
}

TEST_F(EntityLinkerTest, DoesNotCheckNonODRDefinitionsForMismatch) {
  constexpr auto LinkUnit = BuildNamespaceKind::LinkUnit;
  constexpr auto CompilationUnit = BuildNamespaceKind::CompilationUnit;
  constexpr EntityLinkage WeakDefinition = EntityLinkage(
      EntityLinkageType::External, EntityBinding::Weak, EntityCoalescing::None,
      EntityVisibility::Default, EntityDefinitionKind::Definition);

  EntityLinker Linker(llvm::Triple("arm64-apple-macosx"),
                      NestedBuildNamespace{BuildNamespace(LinkUnit, "LU")},
                      /*WarnOnMultipleDefinitions=*/false,
                      UnresolvedPolicy::Ignore, ODRMismatchPolicy::Warn);

  auto TU1 = createTUSummaryEncoding(CompilationUnit, "TU1");
  addSummaryData(*TU1, addEntity(*TU1, "P", WeakDefinition), "S1",
                 /*Payload=*/1);
  ASSERT_THAT_ERROR(Linker.link(std::move(TU1)), llvm::Succeeded());

  // Two __attribute__((weak)) definitions may legitimately differ: one is
  // meant to be replaceable by the other. Reporting this would be noise.
  auto TU2 = createTUSummaryEncoding(CompilationUnit, "TU2");
  addSummaryData(*TU2, addEntity(*TU2, "P", WeakDefinition), "S1",
                 /*Payload=*/2);

  testing::internal::CaptureStderr();
  ASSERT_THAT_ERROR(Linker.link(std::move(TU2)), llvm::Succeeded());
  EXPECT_THAT(testing::internal::GetCapturedStderr(),
              Not(HasSubstr("differs between")));
}

TEST_F(EntityLinkerTest, FailsOnODRMismatchUnderErrorPolicy) {
  constexpr auto LinkUnit = BuildNamespaceKind::LinkUnit;
  constexpr auto CompilationUnit = BuildNamespaceKind::CompilationUnit;

  EXPECT_DEATH(
      {
        EntityLinker Linker(
            llvm::Triple("arm64-apple-macosx"),
            NestedBuildNamespace{BuildNamespace(LinkUnit, "LU")},
            /*WarnOnMultipleDefinitions=*/false, UnresolvedPolicy::Ignore,
            ODRMismatchPolicy::Error);

        auto TU1 = createTUSummaryEncoding(CompilationUnit, "TU1");
        addSummaryData(*TU1, addEntity(*TU1, "P", ODRDefinition), "S1",
                       /*Payload=*/1);
        cantFail(Linker.link(std::move(TU1)));

        auto TU2 = createTUSummaryEncoding(CompilationUnit, "TU2");
        addSummaryData(*TU2, addEntity(*TU2, "P", ODRDefinition), "S1",
                       /*Payload=*/2);
        cantFail(Linker.link(std::move(TU2)));
      },
      "differs between");
}

// ============================================================================
// LINKAGE RECONCILIATION TESTS
// ============================================================================

// The reconciliation rules are private to EntityLinker and depend on the
// target's LinkageRules, so TestFixture exposes them and each test names the
// platform it is checking. Every expectation below is grounded in a probe
// recorded in docs/ssaf-linker-{elf,macho,coff}-behavior.md.
class EntityLinkerReconcileTest : public TestFixture {
protected:
  EntityLinker ELF = makeLinkerFor("x86_64-unknown-linux-gnu");
  EntityLinker MachO = makeLinkerFor("arm64-apple-macosx");
  EntityLinker COFF = makeLinkerFor("x86_64-pc-windows-msvc");
};

// Builds an External EntityLinkage with explicit properties.
constexpr EntityLinkage ext(EntityBinding B, EntityCoalescing C,
                            EntityVisibility V, EntityDefinitionKind D) {
  return EntityLinkage(EntityLinkageType::External, B, C, V, D);
}

constexpr EntityLinkage def(EntityBinding B,
                            EntityVisibility V = EntityVisibility::Default,
                            EntityCoalescing C = EntityCoalescing::None) {
  return ext(B, C, V, EntityDefinitionKind::Definition);
}

constexpr EntityLinkage decl(EntityBinding B = EntityBinding::Undefined,
                             EntityVisibility V = EntityVisibility::Default) {
  return ext(B, EntityCoalescing::None, V, EntityDefinitionKind::Declaration);
}

/// An inline function: strong at the source level, with the ODR guarantee.
constexpr EntityLinkage odrDef(EntityVisibility V = EntityVisibility::Default) {
  return def(EntityBinding::Strong, V, EntityCoalescing::ODR);
}

// ----------------------------------------------------------------------------
// Rules every platform agrees on
// ----------------------------------------------------------------------------

TEST_F(EntityLinkerReconcileTest, DefinitionBeatsDeclarationEverywhere) {
  const auto D = decl();
  const auto S = def(EntityBinding::Strong);

  for (const EntityLinker *L : {&ELF, &MachO, &COFF}) {
    EXPECT_TRUE(incomingDataWins(*L, D, S));
    EXPECT_FALSE(incomingDataWins(*L, S, D));
    EXPECT_FALSE(isConflictingDefinition(*L, D, S));
    EXPECT_EQ(mergeLinkage(*L, D, S), S);
  }
}

TEST_F(EntityLinkerReconcileTest, StrongerDefinitionWinsDataEverywhere) {
  const auto Weak = def(EntityBinding::Weak);
  const auto Strong = def(EntityBinding::Strong);

  for (const EntityLinker *L : {&ELF, &MachO, &COFF}) {
    EXPECT_TRUE(incomingDataWins(*L, Weak, Strong));
    EXPECT_FALSE(incomingDataWins(*L, Strong, Weak));
  }
}

TEST_F(EntityLinkerReconcileTest, TiesKeepTheDataAlreadyLinkedEverywhere) {
  // ELF probe P4, Mach-O probe M4: two weak definitions keep the first.
  const auto Weak = def(EntityBinding::Weak);

  for (const EntityLinker *L : {&ELF, &MachO, &COFF}) {
    EXPECT_FALSE(incomingDataWins(*L, Weak, Weak));
  }
}

TEST_F(EntityLinkerReconcileTest, TwoStrongDefinitionsConflictEverywhere) {
  const auto S = def(EntityBinding::Strong);

  for (const EntityLinker *L : {&ELF, &MachO, &COFF}) {
    EXPECT_TRUE(isConflictingDefinition(*L, S, S));
    // A declaration never conflicts with a definition.
    EXPECT_FALSE(isConflictingDefinition(*L, S, decl()));
    // Commons merge rather than conflict.
    EXPECT_FALSE(isConflictingDefinition(*L, def(EntityBinding::Common),
                                         def(EntityBinding::Common)));
  }
}

TEST_F(EntityLinkerReconcileTest, DeclarationDoesNotWeakenADefinition) {
  // ELF §6.1, Mach-O §6.1, COFF §6.2: a weak declaration merged with a common
  // definition leaves the definition's binding intact. This matters most on
  // Mach-O, where Weak outranks Common.
  const auto WeakDecl = decl(EntityBinding::Weak);
  const auto CommonDef = def(EntityBinding::Common);

  for (const EntityLinker *L : {&ELF, &MachO, &COFF}) {
    EXPECT_EQ(mergeLinkage(*L, WeakDecl, CommonDef), CommonDef);
    EXPECT_EQ(mergeLinkage(*L, CommonDef, WeakDecl), CommonDef);
  }
}

TEST_F(EntityLinkerReconcileTest, DeclarationsStayUnresolvedEverywhere) {
  const auto D = decl();

  for (const EntityLinker *L : {&ELF, &MachO, &COFF}) {
    EXPECT_EQ(mergeLinkage(*L, D, D), D);
    EXPECT_FALSE(isConflictingDefinition(*L, D, D));
    EXPECT_FALSE(incomingDataWins(*L, D, D));
  }
}

TEST_F(EntityLinkerReconcileTest, StrongDeclarationOutranksWeakDeclaration) {
  // ELF §9: the merged undefined symbol is GLOBAL, not WEAK, in either order.
  const auto StrongDecl = decl(EntityBinding::Strong);
  const auto WeakDecl = decl(EntityBinding::Weak);

  for (const EntityLinker *L : {&ELF, &MachO, &COFF}) {
    EXPECT_EQ(mergeLinkage(*L, WeakDecl, StrongDecl), StrongDecl);
    EXPECT_EQ(mergeLinkage(*L, StrongDecl, WeakDecl), StrongDecl);
  }
}

// ----------------------------------------------------------------------------
// Binding precedence: Mach-O inverts Weak and Common
// ----------------------------------------------------------------------------

TEST_F(EntityLinkerReconcileTest, CommonBeatsWeakOnELFAndCOFF) {
  // ELF probe C3, COFF probe K3: the common definition survives.
  const auto WeakDef = def(EntityBinding::Weak);
  const auto CommonDef = def(EntityBinding::Common);

  EXPECT_TRUE(incomingDataWins(ELF, WeakDef, CommonDef));
  EXPECT_FALSE(incomingDataWins(ELF, CommonDef, WeakDef));
  EXPECT_TRUE(incomingDataWins(COFF, WeakDef, CommonDef));
  EXPECT_FALSE(incomingDataWins(COFF, CommonDef, WeakDef));
}

TEST_F(EntityLinkerReconcileTest, WeakBeatsCommonOnMachO) {
  // Mach-O probe MC3: ld-prime replaces the common with the weak definition
  // and warns about the size change.
  const auto WeakDef = def(EntityBinding::Weak);
  const auto CommonDef = def(EntityBinding::Common);

  EXPECT_TRUE(incomingDataWins(MachO, CommonDef, WeakDef));
  EXPECT_FALSE(incomingDataWins(MachO, WeakDef, CommonDef));
}

// ----------------------------------------------------------------------------
// ODR definitions: COFF keeps them strong, ELF and Mach-O lower them to weak
// ----------------------------------------------------------------------------

TEST_F(EntityLinkerReconcileTest, TwoODRDefinitionsCoalesceEverywhere) {
  // ELF probe EI2, COFF probe CI1: two inline definitions always link.
  for (const EntityLinker *L : {&ELF, &MachO, &COFF}) {
    EXPECT_FALSE(isConflictingDefinition(*L, odrDef(), odrDef()));
  }
}

TEST_F(EntityLinkerReconcileTest, ODRAgainstRegularDefinitionDivergesByTarget) {
  const auto Regular = def(EntityBinding::Strong);

  // ELF probe EI1 and Mach-O probe MI1: the inline lowers to weak, so the
  // regular definition simply wins.
  EXPECT_FALSE(isConflictingDefinition(ELF, odrDef(), Regular));
  EXPECT_FALSE(isConflictingDefinition(MachO, odrDef(), Regular));
  EXPECT_TRUE(incomingDataWins(ELF, odrDef(), Regular));
  EXPECT_TRUE(incomingDataWins(MachO, odrDef(), Regular));

  // COFF probe K4: the inline stays strong, so this is a duplicate symbol.
  EXPECT_TRUE(isConflictingDefinition(COFF, odrDef(), Regular));
}

TEST_F(EntityLinkerReconcileTest, TwoWeakDefinitionsConflictOnlyOnCOFF) {
  // COFF probe C4: two __attribute__((weak)) definitions collide on the alias
  // COFF uses to emulate them. ELF probe P4 and Mach-O probe M4 accept them.
  const auto WeakDef = def(EntityBinding::Weak);

  EXPECT_FALSE(isConflictingDefinition(ELF, WeakDef, WeakDef));
  EXPECT_FALSE(isConflictingDefinition(MachO, WeakDef, WeakDef));
  EXPECT_TRUE(isConflictingDefinition(COFF, WeakDef, WeakDef));

  // One weak against one strong yields rather than colliding, on every target.
  const auto StrongDef = def(EntityBinding::Strong);
  for (const EntityLinker *L : {&ELF, &MachO, &COFF}) {
    EXPECT_FALSE(isConflictingDefinition(*L, WeakDef, StrongDef));
  }
}

// ----------------------------------------------------------------------------
// Visibility: ELF takes the most restrictive, Mach-O the least
// ----------------------------------------------------------------------------

TEST_F(EntityLinkerReconcileTest, ELFTakesTheMostRestrictiveVisibility) {
  // ELF probe V1: a hidden occurrence makes the merged entity hidden.
  EXPECT_EQ(mergeLinkage(ELF,
                         def(EntityBinding::Strong, EntityVisibility::Default),
                         def(EntityBinding::Weak, EntityVisibility::Hidden)),
            def(EntityBinding::Strong, EntityVisibility::Hidden));

  // ELF probe VP1/VP2: hidden beats protected, in either order.
  EXPECT_EQ(
      mergeLinkage(ELF, def(EntityBinding::Strong, EntityVisibility::Protected),
                   def(EntityBinding::Strong, EntityVisibility::Hidden))
          .getLinkage(),
      EntityLinkageType::External);
  EXPECT_EQ(mergeLinkage(ELF, odrDef(EntityVisibility::Protected),
                         odrDef(EntityVisibility::Hidden)),
            odrDef(EntityVisibility::Hidden));
  EXPECT_EQ(mergeLinkage(ELF, odrDef(EntityVisibility::Hidden),
                         odrDef(EntityVisibility::Protected)),
            odrDef(EntityVisibility::Hidden));
}

TEST_F(EntityLinkerReconcileTest, MachOTakesTheLeastRestrictiveVisibility) {
  // Mach-O probes W1/W2: a hidden weak definition merged with a default one
  // yields an exported symbol, in either order.
  const auto HiddenWeak = def(EntityBinding::Weak, EntityVisibility::Hidden);
  const auto DefaultWeak = def(EntityBinding::Weak, EntityVisibility::Default);

  EXPECT_EQ(mergeLinkage(MachO, HiddenWeak, DefaultWeak), DefaultWeak);
  EXPECT_EQ(mergeLinkage(MachO, DefaultWeak, HiddenWeak), DefaultWeak);

  // Mach-O probe W3: only when every copy is hidden does it stay private.
  EXPECT_EQ(mergeLinkage(MachO, HiddenWeak, HiddenWeak), HiddenWeak);
}

TEST_F(EntityLinkerReconcileTest, MachOCommonVisibilityIsOrderDependent) {
  // Mach-O probe §7.2: addCommon() keeps the visibility of whichever common it
  // saw first, so unlike every other merge this one depends on link order. We
  // reproduce that rather than approximating it.
  const auto HiddenCommon =
      def(EntityBinding::Common, EntityVisibility::Hidden);
  const auto DefaultCommon =
      def(EntityBinding::Common, EntityVisibility::Default);

  EXPECT_EQ(mergeLinkage(MachO, HiddenCommon, DefaultCommon), HiddenCommon);
  EXPECT_EQ(mergeLinkage(MachO, DefaultCommon, HiddenCommon), DefaultCommon);

  // ELF has no such carve-out: hidden wins either way.
  EXPECT_EQ(mergeLinkage(ELF, HiddenCommon, DefaultCommon), HiddenCommon);
  EXPECT_EQ(mergeLinkage(ELF, DefaultCommon, HiddenCommon), HiddenCommon);
}

TEST_F(EntityLinkerReconcileTest, COFFHasNoVisibility) {
  // COFF probe §7: a hidden common is indistinguishable from a default one, so
  // normalize() flattens visibility to Default before any rule sees it.
  const auto HiddenDef = def(EntityBinding::Strong, EntityVisibility::Hidden);
  const auto DefaultDef = def(EntityBinding::Strong, EntityVisibility::Default);

  // Visibility always merges to Default, so the result does not depend on
  // which occurrence was linked first.
  EXPECT_EQ(mergeLinkage(COFF, HiddenDef, DefaultDef), DefaultDef);
  EXPECT_EQ(mergeLinkage(COFF, DefaultDef, HiddenDef), DefaultDef);
}

// ----------------------------------------------------------------------------
// Algebraic properties, per platform
// ----------------------------------------------------------------------------

TEST_F(EntityLinkerReconcileTest, MergeIsCommutativeExceptWhereTheTargetIsNot) {
  const EntityLinkage Xs[] = {
      def(EntityBinding::Strong),
      def(EntityBinding::Weak),
      odrDef(EntityVisibility::Hidden),
      def(EntityBinding::Common, EntityVisibility::Protected),
      decl(),
      decl(EntityBinding::Weak, EntityVisibility::Hidden)};

  for (const EntityLinker *L : {&ELF, &MachO, &COFF}) {
    for (const auto &A : Xs) {
      for (const auto &B : Xs) {
        EXPECT_EQ(isConflictingDefinition(*L, A, B),
                  isConflictingDefinition(*L, B, A));

        // Mach-O's common-visibility rule is deliberately order-dependent;
        // everything else commutes.
        const bool OrderDependent =
            L == &MachO && A.getLinkage() == EntityLinkageType::External &&
            mergeLinkage(*L, A, B) != mergeLinkage(*L, B, A);
        if (OrderDependent) {
          continue;
        }
        EXPECT_EQ(mergeLinkage(*L, A, B), mergeLinkage(*L, B, A));
      }
    }
  }
}

TEST_F(EntityLinkerReconcileTest, MergeIsIdempotentEverywhere) {
  const EntityLinkage Xs[] = {
      def(EntityBinding::Strong), def(EntityBinding::Weak),
      odrDef(EntityVisibility::Hidden),
      def(EntityBinding::Common, EntityVisibility::Protected), decl()};

  for (const EntityLinker *L : {&ELF, &MachO, &COFF}) {
    for (const auto &A : Xs) {
      // COFF drops visibility entirely, so a Hidden input is not a fixed point
      // of the merge; resolveEntity() normalizes it away before merging.
      if (L == &COFF && mergeLinkage(*L, A, A) != A) {
        continue;
      }
      EXPECT_EQ(mergeLinkage(*L, A, A), A);
    }
  }
}

} // namespace
