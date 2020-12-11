//===- unittests/AST/StmtPrinterTest.cpp --- Statement printer tests ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains tests for Stmt::printPretty() and related methods.
//
// Search this file for WRONG to see test cases that are producing something
// completely wrong, invalid C++ or just misleading.
//
// These tests have a coding convention:
// * statements to be printed should be contained within a function named 'A'
//   unless it should have some special name (e.g., 'operator+');
// * additional helper declarations are 'Z', 'Y', 'X' and so on.
//
//===----------------------------------------------------------------------===//

#include "ASTPrint.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/ADT/SmallString.h"
#include "gtest/gtest.h"

using namespace clang;
using namespace ast_matchers;
using namespace tooling;

namespace {

enum class StdVer { CXX98, CXX11, CXX14, CXX17, CXX2a };

DeclarationMatcher FunctionBodyMatcher(StringRef ContainingFunction) {
  return functionDecl(hasName(ContainingFunction),
                      has(compoundStmt(has(stmt().bind("id")))));
}

template <typename T>
::testing::AssertionResult
PrintedStmtCXXMatches(StdVer Standard, StringRef Code, const T &NodeMatch,
                      StringRef ExpectedPrinted,
                      PolicyAdjusterType PolicyAdjuster = None) {
  const char *StdOpt;
  switch (Standard) {
  case StdVer::CXX98: StdOpt = "-std=c++98"; break;
  case StdVer::CXX11: StdOpt = "-std=c++11"; break;
  case StdVer::CXX14: StdOpt = "-std=c++14"; break;
  case StdVer::CXX17: StdOpt = "-std=c++17"; break;
  case StdVer::CXX2a: StdOpt = "-std=c++2a"; break;
  }

  std::vector<std::string> Args = {
    StdOpt,
    "-Wno-unused-value",
  };
  return PrintedStmtMatches(Code, Args, NodeMatch, ExpectedPrinted,
                            PolicyAdjuster);
}

template <typename T>
::testing::AssertionResult
PrintedStmtMSMatches(StringRef Code, const T &NodeMatch,
                     StringRef ExpectedPrinted,
                     PolicyAdjusterType PolicyAdjuster = None) {
  std::vector<std::string> Args = {
    "-std=c++98",
    "-target", "i686-pc-win32",
    "-fms-extensions",
    "-Wno-unused-value",
  };
  return PrintedStmtMatches(Code, Args, NodeMatch, ExpectedPrinted,
                            PolicyAdjuster);
}

template <typename T>
::testing::AssertionResult
PrintedStmtObjCMatches(StringRef Code, const T &NodeMatch,
                       StringRef ExpectedPrinted,
                       PolicyAdjusterType PolicyAdjuster = None) {
  std::vector<std::string> Args = {
    "-ObjC",
    "-fobjc-runtime=macosx-10.12.0",
  };
  return PrintedStmtMatches(Code, Args, NodeMatch, ExpectedPrinted,
                            PolicyAdjuster);
}

} // unnamed namespace

TEST(StmtPrinter, TestIntegerLiteral) {
  ASSERT_TRUE(PrintedStmtCXXMatches(StdVer::CXX98,
    "void A() {"
    "  1, -1, 1U, 1u,"
    "  1L, 1l, -1L, 1UL, 1ul,"
    "  1LL, -1LL, 1ULL;"
    "}",
    FunctionBodyMatcher("A"),
    "1 , -1 , 1U , 1U , "
    "1L , 1L , -1L , 1UL , 1UL , "
    "1LL , -1LL , 1ULL"));
    // Should be: with semicolon
}

TEST(StmtPrinter, TestMSIntegerLiteral) {
  ASSERT_TRUE(PrintedStmtMSMatches(
    "void A() {"
    "  1i8, -1i8, 1ui8, "
    "  1i16, -1i16, 1ui16, "
    "  1i32, -1i32, 1ui32, "
    "  1i64, -1i64, 1ui64;"
    "}",
    FunctionBodyMatcher("A"),
    "1i8 , -1i8 , 1Ui8 , "
    "1i16 , -1i16 , 1Ui16 , "
    "1 , -1 , 1U , "
    "1LL , -1LL , 1ULL"));
    // Should be: with semicolon
}

TEST(StmtPrinter, TestFloatingPointLiteral) {
  ASSERT_TRUE(PrintedStmtCXXMatches(StdVer::CXX98,
    "void A() { 1.0f, -1.0f, 1.0, -1.0, 1.0l, -1.0l; }",
    FunctionBodyMatcher("A"),
    "1.F , -1.F , 1. , -1. , 1.L , -1.L"));
    // Should be: with semicolon
}

TEST(StmtPrinter, TestCXXConversionDeclImplicit) {
  ASSERT_TRUE(PrintedStmtCXXMatches(
      StdVer::CXX98,
      "struct A {"
      "operator void *();"
      "A operator&(A);"
      "};"
      "void bar(void *);"
      "void foo(A a, A b) {"
      "  bar(a & b);"
      "}",
      traverse(TK_AsIs, cxxMemberCallExpr(anything()).bind("id")), "a & b"));
}

TEST(StmtPrinter, TestCXXConversionDeclExplicit) {
  ASSERT_TRUE(PrintedStmtCXXMatches(StdVer::CXX11,
    "struct A {"
      "operator void *();"
      "A operator&(A);"
    "};"
    "void bar(void *);"
    "void foo(A a, A b) {"
    "  auto x = (a & b).operator void *();"
    "}",
    cxxMemberCallExpr(anything()).bind("id"),
    "(a & b)"));
    // WRONG; Should be: (a & b).operator void *()
}

TEST(StmtPrinter, TestCXXLamda) {
  ASSERT_TRUE(PrintedStmtCXXMatches(StdVer::CXX11,
    "void A() {"
    "  auto l = [] { };"
    "}",
    lambdaExpr(anything()).bind("id"),
    "[] {\n"
    "}"));

  ASSERT_TRUE(PrintedStmtCXXMatches(StdVer::CXX11,
    "void A() {"
    "  int a = 0, b = 1;"
    "  auto l = [a,b](int c, float d) { };"
    "}",
    lambdaExpr(anything()).bind("id"),
    "[a, b](int c, float d) {\n"
    "}"));

  ASSERT_TRUE(PrintedStmtCXXMatches(StdVer::CXX14,
    "void A() {"
    "  auto l = [](auto a, int b, auto c, int, auto) { };"
    "}",
    lambdaExpr(anything()).bind("id"),
    "[](auto a, int b, auto c, int, auto) {\n"
    "}"));

  ASSERT_TRUE(PrintedStmtCXXMatches(StdVer::CXX2a,
    "void A() {"
    "  auto l = []<typename T1, class T2, int I,"
    "              template<class, typename> class T3>"
    "           (int a, auto, int, auto d) { };"
    "}",
    lambdaExpr(anything()).bind("id"),
    "[]<typename T1, class T2, int I, template <class, typename> class T3>(int a, auto, int, auto d) {\n"
    "}"));
}

TEST(StmtPrinter, TestNoImplicitBases) {
  const char *CPPSource = R"(
class A {
  int field;
  int member() { return field; }
};
)";
  // No implicit 'this'.
  ASSERT_TRUE(PrintedStmtCXXMatches(StdVer::CXX11,
      CPPSource, memberExpr(anything()).bind("id"), "field",
      PolicyAdjusterType(
          [](PrintingPolicy &PP) { PP.SuppressImplicitBase = true; })));
  // Print implicit 'this'.
  ASSERT_TRUE(PrintedStmtCXXMatches(StdVer::CXX11,
      CPPSource, memberExpr(anything()).bind("id"), "this->field"));

  const char *ObjCSource = R"(
@interface I {
   int ivar;
}
@end
@implementation I
- (int) method {
  return ivar;
}
@end
      )";
  // No implicit 'self'.
  ASSERT_TRUE(PrintedStmtObjCMatches(ObjCSource, returnStmt().bind("id"),
                                     "return ivar;\n",
                                     PolicyAdjusterType([](PrintingPolicy &PP) {
                                       PP.SuppressImplicitBase = true;
                                     })));
  // Print implicit 'self'.
  ASSERT_TRUE(PrintedStmtObjCMatches(ObjCSource, returnStmt().bind("id"),
                                     "return self->ivar;\n"));
}

TEST(StmtPrinter, TerseOutputWithLambdas) {
  const char *CPPSource = "auto lamb = []{ return 0; };";

  // body is printed when TerseOutput is off(default).
  ASSERT_TRUE(PrintedStmtCXXMatches(StdVer::CXX11, CPPSource,
                                    lambdaExpr(anything()).bind("id"),
                                    "[] {\n    return 0;\n}"));

  // body not printed when TerseOutput is on.
  ASSERT_TRUE(PrintedStmtCXXMatches(
      StdVer::CXX11, CPPSource, lambdaExpr(anything()).bind("id"), "[] {}",
      PolicyAdjusterType([](PrintingPolicy &PP) { PP.TerseOutput = true; })));
}
