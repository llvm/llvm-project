//===- unittests/AST/ProfilingTest.cpp --- Tests for Profiling ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Tooling/Tooling.h"
#include "gtest/gtest.h"
#include <utility>

namespace clang {
namespace {
using namespace ast_matchers;

static auto getClassTemplateRedecls() {
  std::string Code = R"cpp(
    template <class> struct A;
    template <class> struct A;
    template <class> struct A;
  )cpp";
  auto AST = tooling::buildASTFromCode(Code);
  ASTContext &Ctx = AST->getASTContext();

  auto MatchResults = match(classTemplateDecl().bind("id"), Ctx);
  SmallVector<ClassTemplateDecl *, 3> Res;
  for (BoundNodes &N : MatchResults) {
    if (auto *CTD = const_cast<ClassTemplateDecl *>(
            N.getNodeAs<ClassTemplateDecl>("id")))
      Res.push_back(CTD);
  }
  assert(Res.size() == 3);
#ifndef NDEBUG
  for (auto &&I : Res)
    assert(I->getCanonicalDecl() == Res[0]);
#endif
  return std::make_tuple(std::move(AST), Res[1], Res[2]);
}

template <class T> static void testTypeNode(const T *T1, const T *T2) {
  {
    llvm::FoldingSetNodeID ID1, ID2;
    T1->Profile(ID1);
    T2->Profile(ID2);
    ASSERT_NE(ID1, ID2);
  }
  auto *CT1 = cast<T>(T1->getCanonicalTypeInternal());
  auto *CT2 = cast<T>(T2->getCanonicalTypeInternal());
  {
    llvm::FoldingSetNodeID ID1, ID2;
    CT1->Profile(ID1);
    CT2->Profile(ID2);
    ASSERT_EQ(ID1, ID2);
  }
}

TEST(Profiling, DeducedTemplateSpecializationType_Name) {
  auto [AST, CTD1, CTD2] = getClassTemplateRedecls();
  ASTContext &Ctx = AST->getASTContext();

  auto *T1 = cast<DeducedTemplateSpecializationType>(
      Ctx.getDeducedTemplateSpecializationType(
          ElaboratedTypeKeyword::None, TemplateName(CTD1), QualType(), false));
  auto *T2 = cast<DeducedTemplateSpecializationType>(
      Ctx.getDeducedTemplateSpecializationType(
          ElaboratedTypeKeyword::None, TemplateName(CTD2), QualType(), false));
  testTypeNode(T1, T2);
}

} // namespace
} // namespace clang
