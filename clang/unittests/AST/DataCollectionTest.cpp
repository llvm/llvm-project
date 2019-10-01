//===- unittests/AST/DataCollectionTest.cpp -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains tests for the DataCollection module.
//
// They work by hashing the collected data of two nodes and asserting that the
// hash values are equal iff the nodes are considered equal.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/DataCollection.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/StmtVisitor.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Tooling/Tooling.h"
#include "gtest/gtest.h"

using namespace clang;
using namespace tooling;
using namespace ast_matchers;

namespace {
class StmtDataCollector : public ConstStmtVisitor<StmtDataCollector> {
  ASTContext &Context;
  llvm::MD5 &DataConsumer;

  template <class T> void addData(const T &Data) {
    data_collection::addDataToConsumer(DataConsumer, Data);
  }

public:
  StmtDataCollector(const Stmt *S, ASTContext &Context, llvm::MD5 &DataConsumer)
      : Context(Context), DataConsumer(DataConsumer) {
    this->Visit(S);
  }

#define DEF_ADD_DATA(CLASS, CODE)                                              \
  template <class Dummy = void> Dummy Visit##CLASS(const CLASS *S) {           \
    CODE;                                                                      \
    ConstStmtVisitor<StmtDataCollector>::Visit##CLASS(S);                      \
  }

#include "clang/AST/StmtDataCollectors.inc"
};
} // end anonymous namespace

namespace {
struct StmtHashMatch : public MatchFinder::MatchCallback {
  unsigned NumFound;
  llvm::MD5::MD5Result &Hash;
  StmtHashMatch(llvm::MD5::MD5Result &Hash) : NumFound(0), Hash(Hash) {}

  void run(const MatchFinder::MatchResult &Result) override {
    const Stmt *S = Result.Nodes.getNodeAs<Stmt>("id");
    if (!S)
      return;
    ++NumFound;
    if (NumFound > 1)
      return;
    llvm::MD5 MD5;
    StmtDataCollector(S, *Result.Context, MD5);
    MD5.final(Hash);
  }
};
} // end anonymous namespace

static testing::AssertionResult hashStmt(llvm::MD5::MD5Result &Hash,
                                         const StatementMatcher &StmtMatch,
                                         StringRef Code) {
  StmtHashMatch Hasher(Hash);
  MatchFinder Finder;
  Finder.addMatcher(StmtMatch, &Hasher);
  std::unique_ptr<FrontendActionFactory> Factory(
      newFrontendActionFactory(&Finder));
  if (!runToolOnCode(Factory->create(), Code))
    return testing::AssertionFailure()
           << "Parsing error in \"" << Code.str() << "\"";
  if (Hasher.NumFound == 0)
    return testing::AssertionFailure() << "Matcher didn't find any statements";
  if (Hasher.NumFound > 1)
    return testing::AssertionFailure()
           << "Matcher should match only one statement "
              "(found "
           << Hasher.NumFound << ")";
  return testing::AssertionSuccess();
}

static testing::AssertionResult
isStmtHashEqual(const StatementMatcher &StmtMatch, StringRef Code1,
                StringRef Code2) {
  llvm::MD5::MD5Result Hash1, Hash2;
  testing::AssertionResult Result = hashStmt(Hash1, StmtMatch, Code1);
  if (!Result)
    return Result;
  if (!(Result = hashStmt(Hash2, StmtMatch, Code2)))
    return Result;

  return testing::AssertionResult(Hash1 == Hash2);
}

TEST(StmtDataCollector, TestDeclRefExpr) {
  ASSERT_TRUE(isStmtHashEqual(declRefExpr().bind("id"), "int x, r = x;",
                              "int x, r = x;"));
  ASSERT_FALSE(isStmtHashEqual(declRefExpr().bind("id"), "int x, r = x;",
                               "int y, r = y;"));
  ASSERT_FALSE(isStmtHashEqual(declRefExpr().bind("id"), "int x, r = x;",
                               "namespace n { int x, r = x; };"));
}

TEST(StmtDataCollector, TestMemberExpr) {
  ASSERT_TRUE(isStmtHashEqual(memberExpr().bind("id"),
                              "struct { int x; } X; int r = X.x;",
                              "struct { int x; } X; int r = (&X)->x;"));
  ASSERT_TRUE(isStmtHashEqual(memberExpr().bind("id"),
                              "struct { int x; } X; int r = X.x;",
                              "struct { int x; } Y; int r = Y.x;"));
  ASSERT_TRUE(isStmtHashEqual(memberExpr().bind("id"),
                              "struct { int x; } X; int r = X.x;",
                              "struct C { int x; } X; int r = X.C::x;"));
  ASSERT_FALSE(isStmtHashEqual(memberExpr().bind("id"),
                               "struct { int x; } X; int r = X.x;",
                               "struct { int y; } X; int r = X.y;"));
}

TEST(StmtDataCollector, TestIntegerLiteral) {
  ASSERT_TRUE(
      isStmtHashEqual(integerLiteral().bind("id"), "int x = 0;", "int x = 0;"));
  ASSERT_TRUE(
      isStmtHashEqual(integerLiteral().bind("id"), "int x = 0;", "int x =00;"));
  ASSERT_FALSE(
      isStmtHashEqual(integerLiteral().bind("id"), "int x = 0;", "int x = 1;"));
}

TEST(StmtDataCollector, TestFloatingLiteral) {
  ASSERT_TRUE(isStmtHashEqual(floatLiteral().bind("id"), "double x = .0;",
                              "double x = .0;"));
  ASSERT_TRUE(isStmtHashEqual(floatLiteral().bind("id"), "double x = .10;",
                              "double x = .1;"));
  ASSERT_TRUE(isStmtHashEqual(floatLiteral().bind("id"), "double x = .1;",
                              "double x = 1e-1;"));
  ASSERT_FALSE(isStmtHashEqual(floatLiteral().bind("id"), "double x = .0;",
                               "double x = .1;"));
}

TEST(StmtDataCollector, TestStringLiteral) {
  ASSERT_TRUE(isStmtHashEqual(stringLiteral().bind("id"), R"(char x[] = "0";)",
                              R"(char x[] = "0";)"));
  ASSERT_FALSE(isStmtHashEqual(stringLiteral().bind("id"), R"(char x[] = "0";)",
                               R"(char x[] = "1";)"));
}

TEST(StmtDataCollector, TestCXXBoolLiteral) {
  ASSERT_TRUE(isStmtHashEqual(cxxBoolLiteral().bind("id"), "bool x = false;",
                              "bool x = false;"));
  ASSERT_FALSE(isStmtHashEqual(cxxBoolLiteral().bind("id"), "bool x = false;",
                               "bool x = true;"));
}

TEST(StmtDataCollector, TestCharacterLiteral) {
  ASSERT_TRUE(isStmtHashEqual(characterLiteral().bind("id"), "char x = '0';",
                              "char x = '0';"));
  ASSERT_TRUE(isStmtHashEqual(characterLiteral().bind("id"),
                              R"(char x = '\0';)",
                              R"(char x = '\x00';)"));
  ASSERT_FALSE(isStmtHashEqual(characterLiteral().bind("id"), "char x = '0';",
                               "char x = '1';"));
}
