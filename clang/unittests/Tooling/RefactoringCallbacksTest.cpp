//===- unittest/ASTMatchers/RefactoringCallbacksTest.cpp ------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Tooling/RefactoringCallbacks.h"
#include "RewriterTestContext.h"
#include "gtest/gtest.h"

namespace clang {
namespace tooling {

using namespace ast_matchers;

template <typename T>
void expectRewritten(const std::string &Code,
                     const std::string &Expected,
                     const T &AMatcher,
                     RefactoringCallback &Callback) {
  MatchFinder Finder;
  Finder.addMatcher(AMatcher, &Callback);
  OwningPtr<tooling::FrontendActionFactory> Factory(
      tooling::newFrontendActionFactory(&Finder));
  ASSERT_TRUE(tooling::runToolOnCode(Factory->create(), Code))
      << "Parsing error in \"" << Code << "\"";
  RewriterTestContext Context;
  FileID ID = Context.createInMemoryFile("input.cc", Code);
  EXPECT_TRUE(tooling::applyAllReplacements(Callback.getReplacements(),
                                            Context.Rewrite));
  EXPECT_EQ(Expected, Context.getRewrittenText(ID));
}

TEST(RefactoringCallbacksTest, ReplacesStmtsWithString) {
  std::string Code = "void f() { int i = 1; }";
  std::string Expected = "void f() { ; }";
  ReplaceStmtWithText Callback("id", ";");
  expectRewritten(Code, Expected, id("id", declStmt()), Callback);
}

TEST(RefactoringCallbacksTest, ReplacesStmtsInCalledMacros) {
  std::string Code = "#define A void f() { int i = 1; }\nA";
  std::string Expected = "#define A void f() { ; }\nA";
  ReplaceStmtWithText Callback("id", ";");
  expectRewritten(Code, Expected, id("id", declStmt()), Callback);
}

TEST(RefactoringCallbacksTest, IgnoresStmtsInUncalledMacros) {
  std::string Code = "#define A void f() { int i = 1; }";
  std::string Expected = "#define A void f() { int i = 1; }";
  ReplaceStmtWithText Callback("id", ";");
  expectRewritten(Code, Expected, id("id", declStmt()), Callback);
}

TEST(RefactoringCallbacksTest, ReplacesInteger) {
  std::string Code = "void f() { int i = 1; }";
  std::string Expected = "void f() { int i = 2; }";
  ReplaceStmtWithText Callback("id", "2");
  expectRewritten(Code, Expected, id("id", expr(integerLiteral())),
                  Callback);
}

TEST(RefactoringCallbacksTest, ReplacesStmtWithStmt) {
  std::string Code = "void f() { int i = false ? 1 : i * 2; }";
  std::string Expected = "void f() { int i = i * 2; }";
  ReplaceStmtWithStmt Callback("always-false", "should-be");
  expectRewritten(Code, Expected,
      id("always-false", conditionalOperator(
          hasCondition(boolLiteral(equals(false))),
          hasFalseExpression(id("should-be", expr())))),
      Callback);
}

TEST(RefactoringCallbacksTest, ReplacesIfStmt) {
  std::string Code = "bool a; void f() { if (a) f(); else a = true; }";
  std::string Expected = "bool a; void f() { f(); }";
  ReplaceIfStmtWithItsBody Callback("id", true);
  expectRewritten(Code, Expected,
      id("id", ifStmt(
          hasCondition(implicitCastExpr(hasSourceExpression(
              declRefExpr(to(varDecl(hasName("a"))))))))),
      Callback);
}

TEST(RefactoringCallbacksTest, RemovesEntireIfOnEmptyElse) {
  std::string Code = "void f() { if (false) int i = 0; }";
  std::string Expected = "void f() {  }";
  ReplaceIfStmtWithItsBody Callback("id", false);
  expectRewritten(Code, Expected,
      id("id", ifStmt(hasCondition(boolLiteral(equals(false))))),
      Callback);
}

} // end namespace ast_matchers
} // end namespace clang
