//===---- QueryParserTest.cpp - clang-query test --------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "QueryParser.h"
#include "Query.h"
#include "QuerySession.h"
#include "llvm/LineEditor/LineEditor.h"
#include "gtest/gtest.h"

using namespace clang;
using namespace clang::query;

class QueryParserTest : public ::testing::Test {
protected:
  QueryParserTest() : QS(llvm::ArrayRef<std::unique_ptr<ASTUnit>>()) {}
  QueryRef parse(StringRef Code) { return QueryParser::parse(Code, QS); }

  QuerySession QS;
};

TEST_F(QueryParserTest, NoOp) {
  QueryRef Q = parse("");
  EXPECT_TRUE(isa<NoOpQuery>(Q));

  Q = parse("\n");
  EXPECT_TRUE(isa<NoOpQuery>(Q));
}

TEST_F(QueryParserTest, Invalid) {
  QueryRef Q = parse("foo");
  ASSERT_TRUE(isa<InvalidQuery>(Q));
  EXPECT_EQ("unknown command: foo", cast<InvalidQuery>(Q)->ErrStr);
}

TEST_F(QueryParserTest, Help) {
  QueryRef Q = parse("help");
  ASSERT_TRUE(isa<HelpQuery>(Q));

  Q = parse("help me");
  ASSERT_TRUE(isa<InvalidQuery>(Q));
  EXPECT_EQ("unexpected extra input: ' me'", cast<InvalidQuery>(Q)->ErrStr);
}

TEST_F(QueryParserTest, Quit) {
  QueryRef Q = parse("quit");
  ASSERT_TRUE(isa<QuitQuery>(Q));

  Q = parse("q");
  ASSERT_TRUE(isa<QuitQuery>(Q));

  Q = parse("quit me");
  ASSERT_TRUE(isa<InvalidQuery>(Q));
  EXPECT_EQ("unexpected extra input: ' me'", cast<InvalidQuery>(Q)->ErrStr);
}

TEST_F(QueryParserTest, Set) {
  QueryRef Q = parse("set");
  ASSERT_TRUE(isa<InvalidQuery>(Q));
  EXPECT_EQ("expected variable name", cast<InvalidQuery>(Q)->ErrStr);

  Q = parse("set foo bar");
  ASSERT_TRUE(isa<InvalidQuery>(Q));
  EXPECT_EQ("unknown variable: 'foo'", cast<InvalidQuery>(Q)->ErrStr);

  Q = parse("set output");
  ASSERT_TRUE(isa<InvalidQuery>(Q));
  EXPECT_EQ("expected 'diag', 'print' or 'dump', got ''",
            cast<InvalidQuery>(Q)->ErrStr);

  Q = parse("set bind-root true foo");
  ASSERT_TRUE(isa<InvalidQuery>(Q));
  EXPECT_EQ("unexpected extra input: ' foo'", cast<InvalidQuery>(Q)->ErrStr);

  Q = parse("set output foo");
  ASSERT_TRUE(isa<InvalidQuery>(Q));
  EXPECT_EQ("expected 'diag', 'print' or 'dump', got 'foo'",
            cast<InvalidQuery>(Q)->ErrStr);

  Q = parse("set output dump");
  ASSERT_TRUE(isa<SetQuery<OutputKind> >(Q));
  EXPECT_EQ(&QuerySession::OutKind, cast<SetQuery<OutputKind> >(Q)->Var);
  EXPECT_EQ(OK_Dump, cast<SetQuery<OutputKind> >(Q)->Value);

  Q = parse("set bind-root foo");
  ASSERT_TRUE(isa<InvalidQuery>(Q));
  EXPECT_EQ("expected 'true' or 'false', got 'foo'",
            cast<InvalidQuery>(Q)->ErrStr);

  Q = parse("set bind-root true");
  ASSERT_TRUE(isa<SetQuery<bool> >(Q));
  EXPECT_EQ(&QuerySession::BindRoot, cast<SetQuery<bool> >(Q)->Var);
  EXPECT_EQ(true, cast<SetQuery<bool> >(Q)->Value);
}

TEST_F(QueryParserTest, Match) {
  QueryRef Q = parse("match decl()");
  ASSERT_TRUE(isa<MatchQuery>(Q));
  EXPECT_TRUE(cast<MatchQuery>(Q)->Matcher.canConvertTo<Decl>());

  Q = parse("m stmt()");
  ASSERT_TRUE(isa<MatchQuery>(Q));
  EXPECT_TRUE(cast<MatchQuery>(Q)->Matcher.canConvertTo<Stmt>());
}

TEST_F(QueryParserTest, LetUnlet) {
  QueryRef Q = parse("let foo decl()");
  ASSERT_TRUE(isa<LetQuery>(Q));
  EXPECT_EQ("foo", cast<LetQuery>(Q)->Name);
  EXPECT_TRUE(cast<LetQuery>(Q)->Value.isMatcher());
  EXPECT_TRUE(cast<LetQuery>(Q)->Value.getMatcher().hasTypedMatcher<Decl>());

  Q = parse("l foo decl()");
  ASSERT_TRUE(isa<LetQuery>(Q));
  EXPECT_EQ("foo", cast<LetQuery>(Q)->Name);
  EXPECT_TRUE(cast<LetQuery>(Q)->Value.isMatcher());
  EXPECT_TRUE(cast<LetQuery>(Q)->Value.getMatcher().hasTypedMatcher<Decl>());

  Q = parse("let bar \"str\"");
  ASSERT_TRUE(isa<LetQuery>(Q));
  EXPECT_EQ("bar", cast<LetQuery>(Q)->Name);
  EXPECT_TRUE(cast<LetQuery>(Q)->Value.isString());
  EXPECT_EQ("str", cast<LetQuery>(Q)->Value.getString());

  Q = parse("let");
  ASSERT_TRUE(isa<InvalidQuery>(Q));
  EXPECT_EQ("expected variable name", cast<InvalidQuery>(Q)->ErrStr);

  Q = parse("unlet x");
  ASSERT_TRUE(isa<LetQuery>(Q));
  EXPECT_EQ("x", cast<LetQuery>(Q)->Name);
  EXPECT_FALSE(cast<LetQuery>(Q)->Value.hasValue());

  Q = parse("unlet");
  ASSERT_TRUE(isa<InvalidQuery>(Q));
  EXPECT_EQ("expected variable name", cast<InvalidQuery>(Q)->ErrStr);

  Q = parse("unlet x bad_data");
  ASSERT_TRUE(isa<InvalidQuery>(Q));
  EXPECT_EQ("unexpected extra input: ' bad_data'",
            cast<InvalidQuery>(Q)->ErrStr);
}

TEST_F(QueryParserTest, Comment) {
  QueryRef Q = parse("# let foo decl()");
  ASSERT_TRUE(isa<NoOpQuery>(Q));

  Q = parse("let foo decl() # creates a decl() matcher called foo");
  ASSERT_TRUE(isa<LetQuery>(Q));

  Q = parse("set bind-root false # reduce noise");
  ASSERT_TRUE(isa<SetQuery<bool>>(Q));
}

TEST_F(QueryParserTest, Complete) {
  std::vector<llvm::LineEditor::Completion> Comps =
      QueryParser::complete("", 0, QS);
  ASSERT_EQ(6u, Comps.size());
  EXPECT_EQ("help ", Comps[0].TypedText);
  EXPECT_EQ("help", Comps[0].DisplayText);
  EXPECT_EQ("let ", Comps[1].TypedText);
  EXPECT_EQ("let", Comps[1].DisplayText);
  EXPECT_EQ("match ", Comps[2].TypedText);
  EXPECT_EQ("match", Comps[2].DisplayText);
  EXPECT_EQ("quit ", Comps[3].TypedText);
  EXPECT_EQ("quit", Comps[3].DisplayText);
  EXPECT_EQ("set ", Comps[4].TypedText);
  EXPECT_EQ("set", Comps[4].DisplayText);
  EXPECT_EQ("unlet ", Comps[5].TypedText);
  EXPECT_EQ("unlet", Comps[5].DisplayText);

  Comps = QueryParser::complete("set o", 5, QS);
  ASSERT_EQ(1u, Comps.size());
  EXPECT_EQ("utput ", Comps[0].TypedText);
  EXPECT_EQ("output", Comps[0].DisplayText);

  Comps = QueryParser::complete("match while", 11, QS);
  ASSERT_EQ(1u, Comps.size());
  EXPECT_EQ("Stmt(", Comps[0].TypedText);
  EXPECT_EQ("Matcher<Stmt> whileStmt(Matcher<WhileStmt>...)",
            Comps[0].DisplayText);

  Comps = QueryParser::complete("m", 1, QS);
  ASSERT_EQ(1u, Comps.size());
  EXPECT_EQ("atch ", Comps[0].TypedText);
  EXPECT_EQ("match", Comps[0].DisplayText);

  Comps = QueryParser::complete("l", 1, QS);
  ASSERT_EQ(1u, Comps.size());
  EXPECT_EQ("et ", Comps[0].TypedText);
  EXPECT_EQ("let", Comps[0].DisplayText);
}
