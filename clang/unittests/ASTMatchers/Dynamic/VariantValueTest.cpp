//===- VariantValueTest.cpp - VariantValue unit tests ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../ASTMatchersTest.h"
#include "clang/ASTMatchers/Dynamic/VariantValue.h"
#include "gtest/gtest.h"

namespace clang {
namespace ast_matchers {
namespace dynamic {
namespace {

TEST(VariantValueTest, Unsigned) {
  const unsigned kUnsigned = 17;
  VariantValue Value = kUnsigned;

  EXPECT_TRUE(Value.isUnsigned());
  EXPECT_EQ(kUnsigned, Value.getUnsigned());

  EXPECT_TRUE(Value.hasValue());
  EXPECT_FALSE(Value.isString());
  EXPECT_FALSE(Value.isMatcher());
}

TEST(VariantValueTest, String) {
  const StringRef kString = "string";
  VariantValue Value = kString;

  EXPECT_TRUE(Value.isString());
  EXPECT_EQ(kString, Value.getString());
  EXPECT_EQ("String", Value.getTypeAsString());

  EXPECT_TRUE(Value.hasValue());
  EXPECT_FALSE(Value.isUnsigned());
  EXPECT_FALSE(Value.isMatcher());
}

TEST(VariantValueTest, DynTypedMatcher) {
  VariantValue Value = VariantMatcher::SingleMatcher(stmt());

  EXPECT_TRUE(Value.hasValue());
  EXPECT_FALSE(Value.isUnsigned());
  EXPECT_FALSE(Value.isString());

  EXPECT_TRUE(Value.isMatcher());
  EXPECT_FALSE(Value.getMatcher().hasTypedMatcher<Decl>());
  EXPECT_TRUE(Value.getMatcher().hasTypedMatcher<UnaryOperator>());
  EXPECT_EQ("Matcher<Stmt>", Value.getTypeAsString());

  // Can only convert to compatible matchers.
  Value = VariantMatcher::SingleMatcher(recordDecl());
  EXPECT_TRUE(Value.isMatcher());
  EXPECT_TRUE(Value.getMatcher().hasTypedMatcher<Decl>());
  EXPECT_FALSE(Value.getMatcher().hasTypedMatcher<UnaryOperator>());
  EXPECT_EQ("Matcher<Decl>", Value.getTypeAsString());

  Value = VariantMatcher::SingleMatcher(ignoringImpCasts(expr()));
  EXPECT_TRUE(Value.isMatcher());
  EXPECT_FALSE(Value.getMatcher().hasTypedMatcher<Decl>());
  EXPECT_FALSE(Value.getMatcher().hasTypedMatcher<Stmt>());
  EXPECT_TRUE(Value.getMatcher().hasTypedMatcher<Expr>());
  EXPECT_TRUE(Value.getMatcher().hasTypedMatcher<IntegerLiteral>());
  EXPECT_FALSE(Value.getMatcher().hasTypedMatcher<GotoStmt>());
  EXPECT_EQ("Matcher<Expr>", Value.getTypeAsString());
}

TEST(VariantValueTest, Assignment) {
  VariantValue Value = StringRef("A");
  EXPECT_TRUE(Value.isString());
  EXPECT_EQ("A", Value.getString());
  EXPECT_TRUE(Value.hasValue());
  EXPECT_FALSE(Value.isBoolean());
  EXPECT_FALSE(Value.isDouble());
  EXPECT_FALSE(Value.isUnsigned());
  EXPECT_FALSE(Value.isMatcher());
  EXPECT_EQ("String", Value.getTypeAsString());

  Value = VariantMatcher::SingleMatcher(recordDecl());
  EXPECT_TRUE(Value.hasValue());
  EXPECT_FALSE(Value.isBoolean());
  EXPECT_FALSE(Value.isDouble());
  EXPECT_FALSE(Value.isUnsigned());
  EXPECT_FALSE(Value.isString());
  EXPECT_TRUE(Value.isMatcher());
  EXPECT_TRUE(Value.getMatcher().hasTypedMatcher<Decl>());
  EXPECT_FALSE(Value.getMatcher().hasTypedMatcher<UnaryOperator>());
  EXPECT_EQ("Matcher<Decl>", Value.getTypeAsString());

  Value = true;
  EXPECT_TRUE(Value.isBoolean());
  EXPECT_EQ(true, Value.getBoolean());
  EXPECT_TRUE(Value.hasValue());
  EXPECT_FALSE(Value.isUnsigned());
  EXPECT_FALSE(Value.isMatcher());
  EXPECT_FALSE(Value.isString());

  Value = 3.14;
  EXPECT_TRUE(Value.isDouble());
  EXPECT_EQ(3.14, Value.getDouble());
  EXPECT_TRUE(Value.hasValue());
  EXPECT_FALSE(Value.isBoolean());
  EXPECT_FALSE(Value.isUnsigned());
  EXPECT_FALSE(Value.isMatcher());
  EXPECT_FALSE(Value.isString());

  Value = 17;
  EXPECT_TRUE(Value.isUnsigned());
  EXPECT_EQ(17U, Value.getUnsigned());
  EXPECT_FALSE(Value.isBoolean());
  EXPECT_FALSE(Value.isDouble());
  EXPECT_TRUE(Value.hasValue());
  EXPECT_FALSE(Value.isMatcher());
  EXPECT_FALSE(Value.isString());

  Value = VariantValue();
  EXPECT_FALSE(Value.hasValue());
  EXPECT_FALSE(Value.isBoolean());
  EXPECT_FALSE(Value.isDouble());
  EXPECT_FALSE(Value.isUnsigned());
  EXPECT_FALSE(Value.isString());
  EXPECT_FALSE(Value.isMatcher());
  EXPECT_EQ("Nothing", Value.getTypeAsString());
}

TEST(VariantValueTest, ImplicitBool) {
  VariantValue Value;
  bool IfTrue = false;
  if (Value) {
    IfTrue = true;
  }
  EXPECT_FALSE(IfTrue);
  EXPECT_TRUE(!Value);

  Value = StringRef();
  IfTrue = false;
  if (Value) {
    IfTrue = true;
  }
  EXPECT_TRUE(IfTrue);
  EXPECT_FALSE(!Value);
}

TEST(VariantValueTest, Matcher) {
  EXPECT_TRUE(matches("class X {};", VariantValue(VariantMatcher::SingleMatcher(
                                                      recordDecl(hasName("X"))))
                                         .getMatcher()
                                         .getTypedMatcher<Decl>()));
  EXPECT_TRUE(
      matches("int x;", VariantValue(VariantMatcher::SingleMatcher(varDecl()))
                            .getMatcher()
                            .getTypedMatcher<Decl>()));
  EXPECT_TRUE(
      matches("int foo() { return 1 + 1; }",
              VariantValue(VariantMatcher::SingleMatcher(functionDecl()))
                  .getMatcher()
                  .getTypedMatcher<Decl>()));
  // Can't get the wrong matcher.
  EXPECT_FALSE(VariantValue(VariantMatcher::SingleMatcher(varDecl()))
                   .getMatcher()
                   .hasTypedMatcher<Stmt>());
#if !defined(NDEBUG) && GTEST_HAS_DEATH_TEST
  // Trying to get the wrong matcher fails an assertion in Matcher<T>.  We don't
  // do this test when building with MSVC because its debug C runtime prints the
  // assertion failure message as a wide string, which gtest doesn't understand.
  EXPECT_DEATH(VariantValue(VariantMatcher::SingleMatcher(varDecl()))
                   .getMatcher()
                   .getTypedMatcher<Stmt>(),
               "hasTypedMatcher");
#endif

  EXPECT_FALSE(matches(
      "int x;", VariantValue(VariantMatcher::SingleMatcher(functionDecl()))
                    .getMatcher()
                    .getTypedMatcher<Decl>()));
  EXPECT_FALSE(
      matches("int foo() { return 1 + 1; }",
              VariantValue(VariantMatcher::SingleMatcher(declRefExpr()))
                  .getMatcher()
                  .getTypedMatcher<Stmt>()));
}

TEST(VariantValueTest, NodeKind) {
  VariantValue Value = ASTNodeKind::getFromNodeKind<Stmt>();
  EXPECT_TRUE(Value.isNodeKind());
  EXPECT_TRUE(Value.getNodeKind().isSame(ASTNodeKind::getFromNodeKind<Stmt>()));

  Value = ASTNodeKind::getFromNodeKind<CXXMethodDecl>();
  EXPECT_TRUE(Value.isNodeKind());
  EXPECT_TRUE(Value.getNodeKind().isSame(
      ASTNodeKind::getFromNodeKind<CXXMethodDecl>()));

  Value.setNodeKind(ASTNodeKind::getFromNodeKind<PointerType>());
  EXPECT_TRUE(Value.isNodeKind());
  EXPECT_TRUE(
      Value.getNodeKind().isSame(ASTNodeKind::getFromNodeKind<PointerType>()));

  Value = 42;
  EXPECT_TRUE(!Value.isNodeKind());
}

} // end anonymous namespace
} // end namespace dynamic
} // end namespace ast_matchers
} // end namespace clang
