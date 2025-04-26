//===-- DumpASTTests.cpp --------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Annotations.h"
#include "DumpAST.h"
#include "TestTU.h"
#include "clang/AST/ASTTypeTraits.h"
#include "llvm/Support/ScopedPrinter.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace clang {
namespace clangd {
namespace {
using testing::Contains;
using testing::Not;
using testing::SizeIs;

MATCHER_P(withDetail, str, "") { return arg.detail == str; }

TEST(DumpASTTests, BasicInfo) {
  std::pair</*Code=*/std::string, /*Expected=*/std::string> Cases[] = {
      {R"cpp(
float root(int *x) {
  return *x + 1;
}
      )cpp",
       R"(
declaration: Function - root
  type: FunctionProto
    type: Builtin - float
    declaration: ParmVar - x
      type: Pointer
        type: Builtin - int
  statement: Compound
    statement: Return
      expression: ImplicitCast - IntegralToFloating
        expression: BinaryOperator - +
          expression: ImplicitCast - LValueToRValue
            expression: UnaryOperator - *
              expression: ImplicitCast - LValueToRValue
                expression: DeclRef - x
          expression: IntegerLiteral - 1
      )"},
      {R"cpp(
namespace root {
struct S { static const int x = 0; ~S(); };
int y = S::x + root::S().x;
}
      )cpp",
       R"(
declaration: Namespace - root
  declaration: CXXRecord - S
    declaration: Var - x
      type: Qualified - const
        type: Builtin - int
      expression: IntegerLiteral - 0
    declaration: CXXDestructor
      type: Record - S
      type: FunctionProto
        type: Builtin - void
    declaration: CXXConstructor
    declaration: CXXConstructor
  declaration: Var - y
    type: Builtin - int
    expression: ExprWithCleanups
      expression: BinaryOperator - +
        expression: ImplicitCast - LValueToRValue
          expression: DeclRef - x
            specifier: TypeSpec
              type: Record - S
        expression: ImplicitCast - LValueToRValue
          expression: Member - x
            expression: CXXBindTemporary
              expression: CXXTemporaryObject - S
                type: Elaborated
                  specifier: Namespace - root::
                  type: Record - S
      )"},
      {R"cpp(
namespace root {
struct S { static const int x = 0; };
int y = S::x + root::S().x;
}
      )cpp",
       R"(
declaration: Namespace - root
  declaration: CXXRecord - S
    declaration: Var - x
      type: Qualified - const
        type: Builtin - int
      expression: IntegerLiteral - 0
    declaration: CXXConstructor
    declaration: CXXConstructor
    declaration: CXXConstructor
    declaration: CXXDestructor
  declaration: Var - y
    type: Builtin - int
    expression: BinaryOperator - +
      expression: ImplicitCast - LValueToRValue
        expression: DeclRef - x
          specifier: TypeSpec
            type: Record - S
      expression: ImplicitCast - LValueToRValue
        expression: Member - x
          expression: CXXTemporaryObject - S
            type: Elaborated
              specifier: Namespace - root::
              type: Record - S
      )"},
      {R"cpp(
namespace root {
template <typename T> int tmpl() {
  (void)tmpl<unsigned>();
  return T::value;
}
}
      )cpp",
       R"(
declaration: Namespace - root
  declaration: FunctionTemplate - tmpl
    declaration: TemplateTypeParm - T
    declaration: Function - tmpl
      type: FunctionProto
        type: Builtin - int
      statement: Compound
        expression: CStyleCast - ToVoid
          type: Builtin - void
          expression: Call
            expression: ImplicitCast - FunctionToPointerDecay
              expression: DeclRef - tmpl
                template argument: Type
                  type: Builtin - unsigned int
        statement: Return
          expression: DependentScopeDeclRef - value
            specifier: TypeSpec
              type: TemplateTypeParm - T
      )"},
      {R"cpp(
struct Foo { char operator+(int); };
char root = Foo() + 42;
      )cpp",
       R"(
declaration: Var - root
  type: Builtin - char
  expression: ExprWithCleanups
    expression: CXXOperatorCall
      expression: ImplicitCast - FunctionToPointerDecay
        expression: DeclRef - operator+
      expression: MaterializeTemporary - lvalue
        expression: CXXTemporaryObject - Foo
          type: Elaborated
            type: Record - Foo
      expression: IntegerLiteral - 42
      )"},
      {R"cpp(
struct Bar {
  int x;
  int root() const {
    return x;
  }
};
      )cpp",
       R"(
declaration: CXXMethod - root
  type: FunctionProto
    type: Builtin - int
  statement: Compound
    statement: Return
      expression: ImplicitCast - LValueToRValue
        expression: Member - x
          expression: CXXThis - const, implicit
      )"},
  };
  for (const auto &Case : Cases) {
    ParsedAST AST = TestTU::withCode(Case.first).build();
    auto Node = dumpAST(DynTypedNode::create(findUnqualifiedDecl(AST, "root")),
                        AST.getTokens(), AST.getASTContext());
    EXPECT_EQ(llvm::StringRef(Case.second).trim(),
              llvm::StringRef(llvm::to_string(Node)).trim());
  }
}

TEST(DumpASTTests, Range) {
  Annotations Case("$var[[$type[[int]] x]];");
  ParsedAST AST = TestTU::withCode(Case.code()).build();
  auto Node = dumpAST(DynTypedNode::create(findDecl(AST, "x")), AST.getTokens(),
                      AST.getASTContext());
  EXPECT_EQ(Node.range, Case.range("var"));
  ASSERT_THAT(Node.children, SizeIs(1)) << "Expected one child typeloc";
  EXPECT_EQ(Node.children.front().range, Case.range("type"));
}

TEST(DumpASTTests, NoRange) {
  auto TU = TestTU::withHeaderCode("void funcFromHeader();");
  TU.Code = "int varFromSource;";
  ParsedAST AST = TU.build();
  auto Node = dumpAST(
      DynTypedNode::create(*AST.getASTContext().getTranslationUnitDecl()),
      AST.getTokens(), AST.getASTContext());
  ASSERT_THAT(Node.children, Contains(withDetail("varFromSource")));
  ASSERT_THAT(Node.children, Not(Contains(withDetail("funcFromHeader"))));
  EXPECT_THAT(Node.arcana, testing::StartsWith("TranslationUnitDecl "));
  ASSERT_FALSE(Node.range) << "Expected no range for translation unit";
}

TEST(DumpASTTests, Arcana) {
  ParsedAST AST = TestTU::withCode("int x;").build();
  auto Node = dumpAST(DynTypedNode::create(findDecl(AST, "x")), AST.getTokens(),
                      AST.getASTContext());
  EXPECT_THAT(Node.arcana, testing::StartsWith("VarDecl "));
  EXPECT_THAT(Node.arcana, testing::EndsWith(" 'int'"));
  ASSERT_THAT(Node.children, SizeIs(1)) << "Expected one child typeloc";
  EXPECT_THAT(Node.children.front().arcana, testing::StartsWith("QualType "));
}

TEST(DumpASTTests, UnbalancedBraces) {
  // Test that we don't crash while trying to compute a source range for the
  // node whose ending brace is missing, and also that the source range is
  // not empty.
  Annotations Case("/*error-ok*/ $func[[int main() {]]");
  ParsedAST AST = TestTU::withCode(Case.code()).build();
  auto Node = dumpAST(DynTypedNode::create(findDecl(AST, "main")),
                      AST.getTokens(), AST.getASTContext());
  ASSERT_EQ(Node.range, Case.range("func"));
}

} // namespace
} // namespace clangd
} // namespace clang
