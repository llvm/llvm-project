//===- unittest/Tooling/RefactoringCallbacksTest.cpp ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RewriterTestContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Tooling/RefactoringCallbacks.h"
#include "gtest/gtest.h"

namespace clang {
namespace tooling {

using namespace ast_matchers;

template <typename T>
void expectRewritten(const std::string &Code, const std::string &Expected,
                     const T &AMatcher, RefactoringCallback &Callback) {
  std::map<std::string, Replacements> FileToReplace;
  ASTMatchRefactorer Finder(FileToReplace);
  Finder.addMatcher(AMatcher, &Callback);
  std::unique_ptr<tooling::FrontendActionFactory> Factory(
      tooling::newFrontendActionFactory(&Finder));
  ASSERT_TRUE(tooling::runToolOnCode(Factory->create(), Code))
      << "Parsing error in \"" << Code << "\"";
  RewriterTestContext Context;
  FileID ID = Context.createInMemoryFile("input.cc", Code);
  EXPECT_TRUE(tooling::applyAllReplacements(FileToReplace["input.cc"],
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
  expectRewritten(Code, Expected, id("id", expr(integerLiteral())), Callback);
}

TEST(RefactoringCallbacksTest, ReplacesStmtWithStmt) {
  std::string Code = "void f() { int i = false ? 1 : i * 2; }";
  std::string Expected = "void f() { int i = i * 2; }";
  ReplaceStmtWithStmt Callback("always-false", "should-be");
  expectRewritten(
      Code, Expected,
      id("always-false",
         conditionalOperator(hasCondition(cxxBoolLiteral(equals(false))),
                             hasFalseExpression(id("should-be", expr())))),
      Callback);
}

TEST(RefactoringCallbacksTest, ReplacesIfStmt) {
  std::string Code = "bool a; void f() { if (a) f(); else a = true; }";
  std::string Expected = "bool a; void f() { f(); }";
  ReplaceIfStmtWithItsBody Callback("id", true);
  expectRewritten(
      Code, Expected,
      id("id", ifStmt(hasCondition(implicitCastExpr(hasSourceExpression(
                   declRefExpr(to(varDecl(hasName("a"))))))))),
      Callback);
}

TEST(RefactoringCallbacksTest, RemovesEntireIfOnEmptyElse) {
  std::string Code = "void f() { if (false) int i = 0; }";
  std::string Expected = "void f() {  }";
  ReplaceIfStmtWithItsBody Callback("id", false);
  expectRewritten(Code, Expected,
                  id("id", ifStmt(hasCondition(cxxBoolLiteral(equals(false))))),
                  Callback);
}

TEST(RefactoringCallbacksTest, TemplateJustText) {
  std::string Code = "void f() { int i = 1; }";
  std::string Expected = "void f() { FOO }";
  auto Callback = ReplaceNodeWithTemplate::create("id", "FOO");
  EXPECT_FALSE(Callback.takeError());
  expectRewritten(Code, Expected, id("id", declStmt()), **Callback);
}

TEST(RefactoringCallbacksTest, TemplateSimpleSubst) {
  std::string Code = "void f() { int i = 1; }";
  std::string Expected = "void f() { long x = 1; }";
  auto Callback = ReplaceNodeWithTemplate::create("decl", "long x = ${init}");
  EXPECT_FALSE(Callback.takeError());
  expectRewritten(Code, Expected,
                  id("decl", varDecl(hasInitializer(id("init", expr())))),
                  **Callback);
}

TEST(RefactoringCallbacksTest, TemplateLiteral) {
  std::string Code = "void f() { int i = 1; }";
  std::string Expected = "void f() { string x = \"$-1\"; }";
  auto Callback = ReplaceNodeWithTemplate::create("decl",
                                                  "string x = \"$$-${init}\"");
  EXPECT_FALSE(Callback.takeError());
  expectRewritten(Code, Expected,
                  id("decl", varDecl(hasInitializer(id("init", expr())))),
                  **Callback);
}

static void ExpectStringError(const std::string &Expected,
                              llvm::Error E) {
  std::string Found;
  handleAllErrors(std::move(E), [&](const llvm::StringError &SE) {
      llvm::raw_string_ostream Stream(Found);
      SE.log(Stream);
    });
  EXPECT_EQ(Expected, Found);
}

TEST(RefactoringCallbacksTest, TemplateUnterminated) {
  auto Callback = ReplaceNodeWithTemplate::create("decl",
                                                  "string x = \"$$-${init\"");
  ExpectStringError("Unterminated ${...} in replacement template near ${init\"",
                    Callback.takeError());
}

TEST(RefactoringCallbacksTest, TemplateUnknownDollar) {
  auto Callback = ReplaceNodeWithTemplate::create("decl",
                                                  "string x = \"$<");
  ExpectStringError("Invalid $ in replacement template near $<",
                    Callback.takeError());
}


} // end namespace ast_matchers
} // end namespace clang
