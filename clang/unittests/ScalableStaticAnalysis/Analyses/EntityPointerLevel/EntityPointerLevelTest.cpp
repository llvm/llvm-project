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
#include "clang/Tooling/Tooling.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <memory>
#include <vector>

using namespace clang;
using namespace ssaf;
using testing::ElementsAre;

namespace {

static std::vector<unsigned> levelsOf(const DeclPointerLevels &DPLs) {
  std::vector<unsigned> Levels;
  for (const DeclPointerLevel &DPL : DPLs)
    Levels.push_back(DPL.PointerLevel);
  return Levels;
}

// `elaborateHigherDeclPointerLevels` expands a DeclPointerLevel into an
// ascending, exhaustive vector of DeclPointerLevels for the same declaration,
// from the given level up to the maximum pointer level of the declared type.
TEST(EntityPointerLevelTest, ElaborateHigherDeclPointerLevels) {
  std::unique_ptr<ASTUnit> AST =
      tooling::buildASTFromCodeWithArgs(R"cpp(
    int *p;         // one pointer level
    int **q;        // two pointer levels
    int ***r;       // three pointer levels
    int *arr[10];   // array of pointers: two levels (array + pointer)
  )cpp",
                                        {"-Wno-unused"});
  ASSERT_TRUE(AST);
  ASTContext &Ctx = AST->getASTContext();

  // Elaborate the decl named `Name` starting at `StartLevel`, checking that
  // every result shares the input's declaration and is-return flag, and return
  // the produced pointer levels.
  auto elaborate = [&](StringRef Name, unsigned StartLevel) {
    const NamedDecl *ND = findDeclByName(Name, Ctx);
    EXPECT_NE(ND, nullptr) << "decl not found: " << Name.str();
    if (!ND)
      return std::vector<unsigned>{};
    DeclPointerLevels DPLs = elaborateHigherDeclPointerLevels(
        DeclPointerLevel{ND, StartLevel, /*IsReturn=*/false});
    for (const DeclPointerLevel &DPL : DPLs) {
      EXPECT_TRUE(DPL.Decl == ND);
      EXPECT_FALSE(DPL.IsReturn);
    }
    return levelsOf(DPLs);
  };

  EXPECT_THAT(elaborate("p", 1), ElementsAre(1U));         // int*
  EXPECT_THAT(elaborate("q", 1), ElementsAre(1U, 2U));     // int**
  EXPECT_THAT(elaborate("r", 1), ElementsAre(1U, 2U, 3U)); // int***
  EXPECT_THAT(elaborate("arr", 1), ElementsAre(1U, 2U));   // int*[10]
  EXPECT_THAT(elaborate("r", 2), ElementsAre(2U, 3U));
  EXPECT_THAT(elaborate("r", 3), ElementsAre(3U));
}

// For a function entity (IsReturn=true), the maximum pointer level is bounded
// by the return type, whose reference must be stripped first.
TEST(EntityPointerLevelTest, ElaborateHigherDeclPointerLevelsForReturn) {
  std::unique_ptr<ASTUnit> AST =
      tooling::buildASTFromCodeWithArgs(R"cpp(
    int **&refret();  // reference to int**: two pointer levels
    int *valret();    // int*: one pointer level
  )cpp",
                                        {"-Wno-unused"});
  ASSERT_TRUE(AST);
  ASTContext &Ctx = AST->getASTContext();

  auto elaborateReturn = [&](StringRef Name, unsigned StartLevel) {
    const FunctionDecl *FD = findFnByName(Name, Ctx);
    EXPECT_NE(FD, nullptr) << "function not found: " << Name.str();
    if (!FD)
      return std::vector<unsigned>{};
    DeclPointerLevels DPLs = elaborateHigherDeclPointerLevels(
        DeclPointerLevel{FD, StartLevel, /*IsReturn=*/true});
    for (const DeclPointerLevel &DPL : DPLs) {
      EXPECT_TRUE(DPL.Decl == FD);
      EXPECT_TRUE(DPL.IsReturn);
    }
    return levelsOf(DPLs);
  };

  EXPECT_THAT(elaborateReturn("refret", 1), ElementsAre(1U, 2U)); // int**&
  EXPECT_THAT(elaborateReturn("valret", 1), ElementsAre(1U));     // int*
}

} // namespace
