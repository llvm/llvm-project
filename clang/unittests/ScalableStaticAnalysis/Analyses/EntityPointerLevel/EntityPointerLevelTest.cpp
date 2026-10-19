//===- EntityPointerLevelTest.cpp -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/ScalableStaticAnalysis/Analyses/EntityPointerLevel/EntityPointerLevel.h"
#include "FindDecl.h"
#include "clang/AST/Decl.h"
#include "clang/Frontend/ASTUnit.h"
#include "clang/Frontend/SSAFOptions.h"
#include "clang/ScalableStaticAnalysis/Core/Model/BuildNamespace.h"
#include "clang/ScalableStaticAnalysis/Core/TUSummary/TUSummary.h"
#include "clang/ScalableStaticAnalysis/Core/TUSummary/TUSummaryBuilder.h"
#include "clang/ScalableStaticAnalysis/Core/TUSummary/TUSummaryExtractor.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/Support/Error.h"
#include "llvm/Testing/Support/Error.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <memory>
#include <ostream>

using namespace clang;
using namespace ssaf;
using testing::AllOf;
using testing::Each;
using testing::ElementsAre;
using testing::Field;
using testing::Matcher;
using testing::Property;
using testing::UnorderedElementsAre;

namespace clang::ssaf {
// Let gmock print DeclPointerLevels by declaration name.  Found by ADL, so it
// must live in the namespace of `DeclPointerLevel`.
void PrintTo(const DeclPointerLevel &DPL, std::ostream *OS) {
  *OS << "DeclPointerLevel { Decl: '"
      << (DPL.Decl ? DPL.Decl->getNameAsString() : "<null>")
      << "', PointerLevel: " << DPL.PointerLevel
      << ", IsReturn: " << (DPL.IsReturn ? "true" : "false") << " }";
}
} // namespace clang::ssaf

namespace {

/// Matches a DeclPointerLevel at pointer level \p Level.
Matcher<const DeclPointerLevel &> hasPointerLevel(unsigned Level) {
  return Field("PointerLevel", &DeclPointerLevel::PointerLevel, Level);
}

/// Matches a DeclPointerLevel of \p ND, for the entity kind selected by
/// \p IsReturn.
Matcher<const DeclPointerLevel &> isLevelOfDecl(const NamedDecl *ND,
                                                bool IsReturn) {
  return AllOf(Field("Decl", &DeclPointerLevel::Decl, ND),
               Field("IsReturn", &DeclPointerLevel::IsReturn, IsReturn));
}

/// Matches exactly the given pointer levels, in order.
template <typename... Levels> auto hasPointerLevels(Levels... Ls) {
  return ElementsAre(hasPointerLevel(Ls)...);
}

/// Elaborates the DeclPointerLevel of the \p DeclT named \p Name in \p Ctx,
/// starting at pointer level \p StartLevel, and checks that every result
/// belongs to that declaration.
template <typename DeclT = NamedDecl>
DeclPointerLevelVec elaborateFor(ASTContext &Ctx, StringRef Name,
                                 unsigned StartLevel, bool IsReturn = false) {
  const DeclT *ND = findDeclByName<DeclT>(Name, Ctx);
  EXPECT_TRUE(ND) << "decl not found: " << Name.str();
  if (!ND)
    return {};

  DeclPointerLevelVec DPLs = elaborateHigherDeclPointerLevels(
      DeclPointerLevel{ND, StartLevel, IsReturn});
  EXPECT_THAT(DPLs, Each(isLevelOfDecl(ND, IsReturn)));
  return DPLs;
}

/// Matches an EntityPointerLevel with the given entity and pointer level.
auto isEntityPointerLevel(EntityId Id, unsigned Level) {
  return AllOf(
      Property("getEntity", &EntityPointerLevel::getEntity, Id),
      Property("getPointerLevel", &EntityPointerLevel::getPointerLevel, Level));
}

struct EntityPointerLevelTest : testing::Test {
  SSAFOptions Opts;
  TUSummary Summary{
      llvm::Triple("fake-unittest-triple"),
      BuildNamespace(BuildNamespaceKind::CompilationUnit, "Mock.cpp")};
  TUSummaryBuilder Builder{Summary, Opts};
  TUSummaryExtractor Extractor{Builder};
};

// `elaborateHigherDeclPointerLevels` expands a DeclPointerLevel into an
// ascending, exhaustive vector of DeclPointerLevels for the same declaration,
// from the given level up to the maximum pointer level of the declared type.
TEST_F(EntityPointerLevelTest, ElaborateHigherDeclPointerLevels) {
  std::unique_ptr<ASTUnit> AST =
      tooling::buildASTFromCodeWithArgs(R"cpp(
    int *p;         // one pointer level
    int **q;        // two pointer levels
    int ***r;       // three pointer levels
    int *arr[10];   // array of pointers: two levels (array + pointer)
    int mat[3][4];  // 2D array: two levels
    typedef int *IP;
    IP *pp;         // sugar for `int **`: two levels
  )cpp",
                                        {"-Wno-unused"});
  ASSERT_TRUE(AST);
  ASTContext &Ctx = AST->getASTContext();

  EXPECT_THAT(elaborateFor(Ctx, "p", 1), hasPointerLevels(1));       // int*
  EXPECT_THAT(elaborateFor(Ctx, "q", 1), hasPointerLevels(1, 2));    // int**
  EXPECT_THAT(elaborateFor(Ctx, "r", 1), hasPointerLevels(1, 2, 3)); // int***
  EXPECT_THAT(elaborateFor(Ctx, "arr", 1), hasPointerLevels(1, 2));  // int*[10]
  EXPECT_THAT(elaborateFor(Ctx, "mat", 1), hasPointerLevels(1, 2)); // int[3][4]
  EXPECT_THAT(elaborateFor(Ctx, "pp", 1), hasPointerLevels(1, 2));  // IP*
  EXPECT_THAT(elaborateFor(Ctx, "r", 2), hasPointerLevels(2, 3));
  EXPECT_THAT(elaborateFor(Ctx, "r", 3), hasPointerLevels(3));
  // A level beyond the declared type's level count is preserved as is:
  EXPECT_THAT(elaborateFor(Ctx, "p", 2), hasPointerLevels(2));
}

// For a function entity (IsReturn=true), the maximum pointer level is bounded
// by the return type, whose reference must be stripped first.
TEST_F(EntityPointerLevelTest, ElaborateHigherDeclPointerLevelsForReturn) {
  std::unique_ptr<ASTUnit> AST =
      tooling::buildASTFromCodeWithArgs(R"cpp(
    int **&refret();  // reference to int**: two pointer levels
    int *valret();    // int*: one pointer level
  )cpp",
                                        {"-Wno-unused"});
  ASSERT_TRUE(AST);
  ASTContext &Ctx = AST->getASTContext();

  EXPECT_THAT(elaborateFor<FunctionDecl>(Ctx, "refret", 1, /*IsReturn=*/true),
              hasPointerLevels(1, 2)); // int**&
  EXPECT_THAT(elaborateFor<FunctionDecl>(Ctx, "valret", 1, /*IsReturn=*/true),
              hasPointerLevels(1)); // int*
}

TEST_F(EntityPointerLevelTest, ToEntityPointerLevel) {
  std::unique_ptr<ASTUnit> AST =
      tooling::buildASTFromCodeWithArgs(R"cpp(
    int ***p;
    int **&refret();
  )cpp",
                                        {"-Wno-unused"});
  ASSERT_TRUE(AST);
  ASTContext &Ctx = AST->getASTContext();

  const NamedDecl *ND = findDeclByName("p", Ctx);
  ASSERT_TRUE(ND);
  std::optional<EntityId> PId = Extractor.addEntity(ND);
  ASSERT_TRUE(PId);

  const FunctionDecl *FD = findFnByName("refret", Ctx);
  ASSERT_TRUE(FD);
  std::optional<EntityId> RefretId = Extractor.addEntityForReturn(FD);
  ASSERT_TRUE(RefretId);

  DeclPointerLevelVec DPLs =
      elaborateHigherDeclPointerLevels({ND, 1, /*IsReturn=*/false});
  DeclPointerLevelVec ReturnDPLs =
      elaborateHigherDeclPointerLevels({FD, 1, /*IsReturn=*/true});
  DPLs.append(ReturnDPLs.begin(), ReturnDPLs.end());
  DPLs.push_back(DPLs.front()); // duplicate, to exercise de-duplication

  ASSERT_THAT_EXPECTED(
      toEntityPointerLevels(DPLs, Ctx, Extractor),
      llvm::HasValue(UnorderedElementsAre(
          isEntityPointerLevel(*PId, 1), isEntityPointerLevel(*PId, 2),
          isEntityPointerLevel(*PId, 3), isEntityPointerLevel(*RefretId, 1),
          isEntityPointerLevel(*RefretId, 2))));
}

} // namespace
