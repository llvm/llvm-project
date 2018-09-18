//===---- QueryTest.cpp - clang-query test --------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "Query.h"
#include "QueryParser.h"
#include "QuerySession.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/ASTMatchers/Dynamic/VariantValue.h"
#include "clang/Frontend/ASTUnit.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"
#include <string>

using namespace clang;
using namespace clang::ast_matchers;
using namespace clang::ast_matchers::dynamic;
using namespace clang::query;
using namespace clang::tooling;

class QueryEngineTest : public ::testing::Test {
  ArrayRef<std::unique_ptr<ASTUnit>> mkASTUnit2(std::unique_ptr<ASTUnit> a,
                                                std::unique_ptr<ASTUnit> b) {
    ASTs[0] = std::move(a);
    ASTs[1] = std::move(b);
    return ArrayRef<std::unique_ptr<ASTUnit>>(ASTs);
  }

protected:
  QueryEngineTest()
      : S(mkASTUnit2(buildASTFromCode("void foo1(void) {}\nvoid foo2(void) {}",
                                      "foo.cc"),
                     buildASTFromCode("void bar1(void) {}\nvoid bar2(void) {}",
                                      "bar.cc"))),
        OS(Str) {}

  std::unique_ptr<ASTUnit> ASTs[2];
  QuerySession S;

  std::string Str;
  llvm::raw_string_ostream OS;
};

TEST_F(QueryEngineTest, Basic) {
  DynTypedMatcher FnMatcher = functionDecl();
  DynTypedMatcher FooMatcher = functionDecl(hasName("foo1"));

  EXPECT_TRUE(NoOpQuery().run(OS, S));

  EXPECT_EQ("", OS.str());

  Str.clear();

  EXPECT_FALSE(InvalidQuery("Parse error").run(OS, S));

  EXPECT_EQ("Parse error\n", OS.str());

  Str.clear();

  EXPECT_TRUE(HelpQuery().run(OS, S));

  EXPECT_TRUE(OS.str().find("Available commands:") != std::string::npos);

  Str.clear();

  EXPECT_TRUE(MatchQuery(FnMatcher).run(OS, S));

  EXPECT_TRUE(OS.str().find("foo.cc:1:1: note: \"root\" binds here") !=
              std::string::npos);
  EXPECT_TRUE(OS.str().find("foo.cc:2:1: note: \"root\" binds here") !=
              std::string::npos);
  EXPECT_TRUE(OS.str().find("bar.cc:1:1: note: \"root\" binds here") !=
              std::string::npos);
  EXPECT_TRUE(OS.str().find("bar.cc:2:1: note: \"root\" binds here") !=
              std::string::npos);
  EXPECT_TRUE(OS.str().find("4 matches.") != std::string::npos);

  Str.clear();

  EXPECT_TRUE(MatchQuery(FooMatcher).run(OS, S));

  EXPECT_TRUE(OS.str().find("foo.cc:1:1: note: \"root\" binds here") !=
              std::string::npos);
  EXPECT_TRUE(OS.str().find("1 match.") != std::string::npos);

  Str.clear();

  EXPECT_TRUE(
      SetQuery<OutputKind>(&QuerySession::OutKind, OK_Print).run(OS, S));
  EXPECT_TRUE(MatchQuery(FooMatcher).run(OS, S));

  EXPECT_TRUE(OS.str().find("Binding for \"root\":\nvoid foo1()") !=
              std::string::npos);

  Str.clear();

  EXPECT_TRUE(SetQuery<OutputKind>(&QuerySession::OutKind, OK_Dump).run(OS, S));
  EXPECT_TRUE(MatchQuery(FooMatcher).run(OS, S));

  EXPECT_TRUE(OS.str().find("FunctionDecl") != std::string::npos);

  Str.clear();

  EXPECT_TRUE(SetQuery<bool>(&QuerySession::BindRoot, false).run(OS, S));
  EXPECT_TRUE(MatchQuery(FooMatcher).run(OS, S));

  EXPECT_TRUE(OS.str().find("No bindings.") != std::string::npos);

  Str.clear();

  EXPECT_FALSE(MatchQuery(isMain()).run(OS, S));

  EXPECT_EQ("Not a valid top-level matcher.\n", OS.str());
}

TEST_F(QueryEngineTest, LetAndMatch) {
  EXPECT_TRUE(QueryParser::parse("let x \"foo1\"", S)->run(OS, S));
  EXPECT_EQ("", OS.str());
  Str.clear();

  EXPECT_TRUE(QueryParser::parse("let y hasName(x)", S)->run(OS, S));
  EXPECT_EQ("", OS.str());
  Str.clear();

  EXPECT_TRUE(QueryParser::parse("match functionDecl(y)", S)->run(OS, S));
  EXPECT_TRUE(OS.str().find("foo.cc:1:1: note: \"root\" binds here") !=
              std::string::npos);
  EXPECT_TRUE(OS.str().find("1 match.") != std::string::npos);
  Str.clear();

  EXPECT_TRUE(QueryParser::parse("unlet x", S)->run(OS, S));
  EXPECT_EQ("", OS.str());
  Str.clear();

  EXPECT_FALSE(QueryParser::parse("let y hasName(x)", S)->run(OS, S));
  EXPECT_EQ("1:2: Error parsing argument 1 for matcher hasName.\n"
            "1:10: Value not found: x\n", OS.str());
  Str.clear();
}
