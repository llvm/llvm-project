//== unittests/Sema/LifetimeBoundImplicitThisTest.cpp ----------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/AST/ASTContext.h"
#include "clang/AST/DeclCXX.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Frontend/ASTUnit.h"
#include "clang/Tooling/Tooling.h"
#include "gtest/gtest.h"
#include <string>

using namespace clang;
using namespace clang::ast_matchers;
using namespace clang::tooling;

namespace {

// The lifetime-safety inference rewrites the method's TypeSourceInfo after
// inferring [[clang::lifetimebound]] on the implicit 'this'.
// The method's type is already the deduced one while the written
// TypeSourceInfo still holds the 'auto' placeholder.
struct TestCase {
  const char *Name;
  const char *Code;
  bool IsInstantiation;
  // The spelling of the written return type the TypeLoc must still point at.
  StringRef WrittenReturnType;
  unsigned NumParams;
};

const TestCase Cases[] = {
    {"NonTemplate", R"cpp(
struct S;
struct Iter {
  S *s;
  explicit Iter(S &x) : s(&x) {}
};
struct S {
  auto make() { return Iter(*this); }
};
auto test() { S b; return b.make(); }
)cpp",
     /*IsInstantiation=*/false, "auto", /*NumParams=*/0},
    {"TemplateInstantiation", R"cpp(
template <class C> struct Iter {
  C *c;
  explicit Iter(C &x) : c(&x) {}
};
template <class T> struct S {
  auto make() { return Iter<S>(*this); }
};
auto test() { S<char> b; return b.make(); }
)cpp",
     /*IsInstantiation=*/true, "auto", /*NumParams=*/0},
    // The parameters must survive the rewrite untouched.
    {"WithParam", R"cpp(
struct S;
struct Iter {
  S *s;
  explicit Iter(S &x) : s(&x) {}
};
struct S {
  auto make(int x) { return Iter(*this); }
};
auto test() { S b; return b.make(0); }
)cpp",
     /*IsInstantiation=*/false, "auto", /*NumParams=*/1},
    // The 'const' qualifier lives on the function type that gets wrapped.
    {"ConstQualified", R"cpp(
struct S;
struct Iter {
  const S *s;
  explicit Iter(const S &x) : s(&x) {}
};
struct S {
  auto make() const { return Iter(*this); }
};
auto test() { S b; return b.make(); }
)cpp",
     /*IsInstantiation=*/false, "auto", /*NumParams=*/0},
    {"DecltypeAuto", R"cpp(
struct S;
struct Iter {
  S *s;
  explicit Iter(S &x) : s(&x) {}
};
struct S {
  decltype(auto) make() { return Iter(*this); }
};
auto test() { S b; return b.make(); }
)cpp",
     /*IsInstantiation=*/false, "decltype(auto)", /*NumParams=*/0},
};

class LifetimeBoundImplicitThisTest : public testing::TestWithParam<TestCase> {
};

// Checks that the inference annotated 'make' and that the return type's
// location still points at the written return type.
TEST_P(LifetimeBoundImplicitThisTest, DeducedReturnTypeTypeLoc) {
  const TestCase &C = GetParam();
  std::unique_ptr<ASTUnit> AST = buildASTFromCodeWithArgs(
      C.Code,
      {"-std=c++17", "-fsyntax-only", "-Xclang", "-flifetime-safety-inference",
       "-Xclang", "-fexperimental-lifetime-safety-tu-analysis",
       "-Wlifetime-safety-suggestions", "-Wlifetime-safety",
       "-Wlifetime-safety-experimental-field-origin-escape", "-Wno-dangling"},
      "input.cc");
  ASSERT_TRUE(AST);
  ASTContext &Ctx = AST->getASTContext();
  const SourceManager &SM = Ctx.getSourceManager();

  auto Matches =
      match(cxxMethodDecl(hasName("make"), isDefinition(),
                          C.IsInstantiation ? isInstantiated()
                                            : unless(isInstantiated()))
                .bind("m"),
            Ctx);
  ASSERT_EQ(Matches.size(), 1u);
  const auto *MD = Matches[0].getNodeAs<CXXMethodDecl>("m");
  ASSERT_TRUE(MD);

  // Inference must have wrapped the function type in an AttributedType.
  EXPECT_TRUE(isa<AttributedType>(MD->getType()));
  EXPECT_TRUE(MD->getTypeSourceInfo()->getTypeLoc().getAs<AttributedTypeLoc>());

  SourceLocation RetBegin = MD->getReturnTypeSourceRange().getBegin();
  ASSERT_TRUE(RetBegin.isValid());
  EXPECT_EQ(
      StringRef(SM.getCharacterData(RetBegin), C.WrittenReturnType.size()),
      C.WrittenReturnType);

  FunctionTypeLoc FTL = MD->getFunctionTypeLoc();
  ASSERT_TRUE(FTL);
  ASSERT_EQ(FTL.getNumParams(), C.NumParams);
  for (unsigned I = 0; I != C.NumParams; ++I)
    EXPECT_EQ(FTL.getParam(I), MD->getParamDecl(I));
}

INSTANTIATE_TEST_SUITE_P(DeducedReturnTypes, LifetimeBoundImplicitThisTest,
                         testing::ValuesIn(Cases),
                         [](const testing::TestParamInfo<TestCase> &Info) {
                           return std::string(Info.param.Name);
                         });

} // namespace
