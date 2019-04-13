//===--- PrintASTTests.cpp ----------------------------------------- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AST.h"
#include "Annotations.h"
#include "Protocol.h"
#include "SourceCode.h"
#include "TestTU.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "gmock/gmock.h"
#include "gtest/gtest-param-test.h"
#include "gtest/gtest.h"
#include "gtest/internal/gtest-param-util-generated.h"

namespace clang {
namespace clangd {
namespace {

using testing::ElementsAreArray;

struct Case {
  const char *AnnotatedCode;
  std::vector<const char *> Expected;
};
class ASTUtils : public testing::Test,
                 public ::testing::WithParamInterface<Case> {};

TEST_P(ASTUtils, PrintTemplateArgs) {
  auto Pair = GetParam();
  Annotations Test(Pair.AnnotatedCode);
  auto AST = TestTU::withCode(Test.code()).build();
  struct Visitor : RecursiveASTVisitor<Visitor> {
    Visitor(std::vector<Position> Points) : Points(std::move(Points)) {}
    bool VisitNamedDecl(const NamedDecl *ND) {
      if (TemplateArgsAtPoints.size() == Points.size())
        return true;
      auto Pos = sourceLocToPosition(ND->getASTContext().getSourceManager(),
                                     ND->getLocation());
      if (Pos != Points[TemplateArgsAtPoints.size()])
        return true;
      TemplateArgsAtPoints.push_back(printTemplateSpecializationArgs(*ND));
      return true;
    }
    std::vector<std::string> TemplateArgsAtPoints;
    const std::vector<Position> Points;
  };
  Visitor V(Test.points());
  V.TraverseDecl(AST.getASTContext().getTranslationUnitDecl());
  EXPECT_THAT(V.TemplateArgsAtPoints, ElementsAreArray(Pair.Expected));
}

INSTANTIATE_TEST_CASE_P(ASTUtilsTests, ASTUtils,
                        testing::ValuesIn(std::vector<Case>({
                            {
                                R"cpp(
                                  template <class X> class Bar {};
                                  template <> class ^Bar<double> {};)cpp",
                                {"<double>"}},
                            {
                                R"cpp(
                                  template <class X> class Bar {};
                                  template <class T, class U,
                                  template<typename> class Z, int Q>
                                  struct Foo {};
                                  template struct ^Foo<int, bool, Bar, 8>;
                                  template <typename T>
                                  struct ^Foo<T *, T, Bar, 3> {};)cpp",
                                {"<int, bool, Bar, 8>", "<T *, T, Bar, 3>"}},
                            {
                                R"cpp(
                                  template <int ...> void Foz() {};
                                  template <> void ^Foz<3, 5, 8>() {};)cpp",
                                {"<3, 5, 8>"}},
                            {
                                R"cpp(
                                  template <class X> class Bar {};
                                  template <template <class> class ...>
                                  class Aux {};
                                  template <> class ^Aux<Bar, Bar> {};
                                  template <template <class> T>
                                  class ^Aux<T, T> {};)cpp",
                                {"<Bar, Bar>", "<T, T>"}},
                            {
                                R"cpp(
                                  template <typename T> T var = 1234;
                                  template <> int ^var<int> = 1;)cpp",
                                {"<int>"}},
                            {
                                R"cpp(
                                  template <typename T> struct Foo;
                                  struct Bar { friend class Foo<int>; };
                                  template <> struct ^Foo<int> {};)cpp",
                                {"<int>"}},
                        })));
} // namespace
} // namespace clangd
} // namespace clang
