//===---- QueryParserTest.cpp - clang-query test --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "QueryParser.h"
#include "Query.h"
#include "QuerySession.h"
#include "clang/Tooling/NodeIntrospection.h"
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

  bool HasIntrospection = tooling::NodeIntrospection::hasIntrospectionSupport();
  QueryRef Q = parse("set");
  ASSERT_TRUE(isa<InvalidQuery>(Q));
  EXPECT_EQ("expected variable name", cast<InvalidQuery>(Q)->ErrStr);

  Q = parse("set foo bar");
  ASSERT_TRUE(isa<InvalidQuery>(Q));
  EXPECT_EQ("unknown variable: 'foo'", cast<InvalidQuery>(Q)->ErrStr);

  Q = parse("set output");
  ASSERT_TRUE(isa<InvalidQuery>(Q));
  if (HasIntrospection)
    EXPECT_EQ(
        "expected 'diag', 'print', 'detailed-ast', 'srcloc' or 'dump', got ''",
        cast<InvalidQuery>(Q)->ErrStr);
  else
    EXPECT_EQ("expected 'diag', 'print', 'detailed-ast' or 'dump', got ''",
              cast<InvalidQuery>(Q)->ErrStr);

  Q = parse("set bind-root true foo");
  ASSERT_TRUE(isa<InvalidQuery>(Q));
  EXPECT_EQ("unexpected extra input: ' foo'", cast<InvalidQuery>(Q)->ErrStr);

  Q = parse("set output foo");
  ASSERT_TRUE(isa<InvalidQuery>(Q));
  if (HasIntrospection)
    EXPECT_EQ("expected 'diag', 'print', 'detailed-ast', 'srcloc' or 'dump', "
              "got 'foo'",
              cast<InvalidQuery>(Q)->ErrStr);
  else
    EXPECT_EQ("expected 'diag', 'print', 'detailed-ast' or 'dump', got 'foo'",
              cast<InvalidQuery>(Q)->ErrStr);

  Q = parse("set output dump");
  ASSERT_TRUE(isa<SetExclusiveOutputQuery >(Q));
  EXPECT_EQ(&QuerySession::DetailedASTOutput, cast<SetExclusiveOutputQuery>(Q)->Var);

  Q = parse("set output detailed-ast");
  ASSERT_TRUE(isa<SetExclusiveOutputQuery>(Q));
  EXPECT_EQ(&QuerySession::DetailedASTOutput, cast<SetExclusiveOutputQuery>(Q)->Var);

  Q = parse("enable output detailed-ast");
  ASSERT_TRUE(isa<EnableOutputQuery>(Q));
  EXPECT_EQ(&QuerySession::DetailedASTOutput, cast<EnableOutputQuery>(Q)->Var);

  Q = parse("enable");
  ASSERT_TRUE(isa<InvalidQuery>(Q));
  EXPECT_EQ("expected variable name", cast<InvalidQuery>(Q)->ErrStr);

  Q = parse("disable output detailed-ast");
  ASSERT_TRUE(isa<DisableOutputQuery>(Q));
  EXPECT_EQ(&QuerySession::DetailedASTOutput, cast<DisableOutputQuery>(Q)->Var);

  Q = parse("set bind-root foo");
  ASSERT_TRUE(isa<InvalidQuery>(Q));
  EXPECT_EQ("expected 'true' or 'false', got 'foo'",
            cast<InvalidQuery>(Q)->ErrStr);

  Q = parse("set bind-root true");
  ASSERT_TRUE(isa<SetQuery<bool> >(Q));
  EXPECT_EQ(&QuerySession::BindRoot, cast<SetQuery<bool> >(Q)->Var);
  EXPECT_EQ(true, cast<SetQuery<bool> >(Q)->Value);

  Q = parse("set traversal AsIs");
  ASSERT_TRUE(isa<SetQuery<TraversalKind>>(Q));
  EXPECT_EQ(&QuerySession::TK, cast<SetQuery<TraversalKind>>(Q)->Var);
  EXPECT_EQ(TK_AsIs, cast<SetQuery<TraversalKind>>(Q)->Value);

  Q = parse("set traversal NotATraversal");
  ASSERT_TRUE(isa<InvalidQuery>(Q));
  EXPECT_EQ("expected traversal kind, got 'NotATraversal'",
            cast<InvalidQuery>(Q)->ErrStr);
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
  ASSERT_EQ(8u, Comps.size());
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
  EXPECT_EQ("enable ", Comps[5].TypedText);
  EXPECT_EQ("enable", Comps[5].DisplayText);
  EXPECT_EQ("disable ", Comps[6].TypedText);
  EXPECT_EQ("disable", Comps[6].DisplayText);
  EXPECT_EQ("unlet ", Comps[7].TypedText);
  EXPECT_EQ("unlet", Comps[7].DisplayText);

  Comps = QueryParser::complete("set o", 5, QS);
  ASSERT_EQ(1u, Comps.size());
  EXPECT_EQ("utput ", Comps[0].TypedText);
  EXPECT_EQ("output", Comps[0].DisplayText);

  Comps = QueryParser::complete("set t", 5, QS);
  ASSERT_EQ(1u, Comps.size());
  EXPECT_EQ("raversal ", Comps[0].TypedText);
  EXPECT_EQ("traversal", Comps[0].DisplayText);

  Comps = QueryParser::complete("enable ", 7, QS);
  ASSERT_EQ(1u, Comps.size());
  EXPECT_EQ("output ", Comps[0].TypedText);
  EXPECT_EQ("output", Comps[0].DisplayText);

  bool HasIntrospection = tooling::NodeIntrospection::hasIntrospectionSupport();

  Comps = QueryParser::complete("enable output ", 14, QS);
  ASSERT_EQ(HasIntrospection ? 5u : 4u, Comps.size());

  EXPECT_EQ("diag ", Comps[0].TypedText);
  EXPECT_EQ("diag", Comps[0].DisplayText);
  EXPECT_EQ("print ", Comps[1].TypedText);
  EXPECT_EQ("print", Comps[1].DisplayText);
  EXPECT_EQ("detailed-ast ", Comps[2].TypedText);
  EXPECT_EQ("detailed-ast", Comps[2].DisplayText);
  if (HasIntrospection) {
    EXPECT_EQ("srcloc ", Comps[3].TypedText);
    EXPECT_EQ("srcloc", Comps[3].DisplayText);
  }
  EXPECT_EQ("dump ", Comps[HasIntrospection ? 4 : 3].TypedText);
  EXPECT_EQ("dump", Comps[HasIntrospection ? 4 : 3].DisplayText);

  Comps = QueryParser::complete("set traversal ", 14, QS);
  ASSERT_EQ(2u, Comps.size());

  EXPECT_EQ("AsIs ", Comps[0].TypedText);
  EXPECT_EQ("AsIs", Comps[0].DisplayText);
  EXPECT_EQ("IgnoreUnlessSpelledInSource ", Comps[1].TypedText);
  EXPECT_EQ("IgnoreUnlessSpelledInSource", Comps[1].DisplayText);

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

TEST_F(QueryParserTest, Multiline) {

  // Single string with multiple commands
  QueryRef Q = parse(R"matcher(
set bind-root false
set output dump
    )matcher");

  ASSERT_TRUE(isa<SetQuery<bool>>(Q));

  Q = parse(Q->RemainingContent);
  ASSERT_TRUE(isa<SetExclusiveOutputQuery>(Q));

  // Missing newline
  Q = parse(R"matcher(
set bind-root false set output dump
    )matcher");

  ASSERT_TRUE(isa<InvalidQuery>(Q));
  EXPECT_EQ("unexpected extra input: ' set output dump\n    '",
            cast<InvalidQuery>(Q)->ErrStr);

  // Commands which do their own parsing
  Q = parse(R"matcher(
let fn functionDecl(hasName("foo"))
match callExpr(callee(functionDecl()))
    )matcher");

  ASSERT_TRUE(isa<LetQuery>(Q));

  Q = parse(Q->RemainingContent);
  ASSERT_TRUE(isa<MatchQuery>(Q));

  // Multi-line matcher
  Q = parse(R"matcher(
match callExpr(callee(
    functionDecl().bind("fn")
    ))

    )matcher");

  ASSERT_TRUE(isa<MatchQuery>(Q));

  // Comment locations
  Q = parse(R"matcher(
#nospacecomment
# Leading comment
match callExpr ( # Trailing comment
            # Comment alone on line

            callee(
            functionDecl(
            ).bind(
            "fn"
            )
            )) # Comment trailing close
# Comment after match
    )matcher");

  ASSERT_TRUE(isa<MatchQuery>(Q));

  // \r\n
  Q = parse("set bind-root false\r\nset output dump");

  ASSERT_TRUE(isa<SetQuery<bool>>(Q));

  Q = parse(Q->RemainingContent);
  ASSERT_TRUE(isa<SetExclusiveOutputQuery>(Q));

  // Leading and trailing space in lines
  Q = parse("  set bind-root false  \r\n  set output dump  ");

  ASSERT_TRUE(isa<SetQuery<bool>>(Q));

  Q = parse(Q->RemainingContent);
  ASSERT_TRUE(isa<SetExclusiveOutputQuery>(Q));

  // Incomplete commands
  Q = parse("set\nbind-root false");

  ASSERT_TRUE(isa<InvalidQuery>(Q));
  EXPECT_EQ("expected variable name", cast<InvalidQuery>(Q)->ErrStr);

  Q = parse("set bind-root\nfalse");

  ASSERT_TRUE(isa<InvalidQuery>(Q));
  EXPECT_EQ("expected 'true' or 'false', got ''",
            cast<InvalidQuery>(Q)->ErrStr);

  Q = parse(R"matcher(
match callExpr
(
)
    )matcher");

  ASSERT_TRUE(isa<InvalidQuery>(Q));
  EXPECT_EQ("1:9: Error parsing matcher. Found token <NewLine> "
            "while looking for '('.",
            cast<InvalidQuery>(Q)->ErrStr);

  Q = parse("let someMatcher\nm parmVarDecl()");

  ASSERT_TRUE(isa<InvalidQuery>(Q));
  EXPECT_EQ("1:1: Invalid token <NewLine> found when looking for a value.",
            cast<InvalidQuery>(Q)->ErrStr);

  Q = parse("\nm parmVarDecl()\nlet someMatcher\nm parmVarDecl()");

  ASSERT_TRUE(isa<MatchQuery>(Q));
  Q = parse(Q->RemainingContent);

  ASSERT_TRUE(isa<InvalidQuery>(Q));
  EXPECT_EQ("1:1: Invalid token <NewLine> found when looking for a value.",
            cast<InvalidQuery>(Q)->ErrStr);

  Q = parse("\nlet someMatcher\n");

  ASSERT_TRUE(isa<InvalidQuery>(Q));
  EXPECT_EQ("1:1: Invalid token <NewLine> found when looking for a value.",
            cast<InvalidQuery>(Q)->ErrStr);

  Q = parse("\nm parmVarDecl()\nlet someMatcher\n");

  ASSERT_TRUE(isa<MatchQuery>(Q));
  Q = parse(Q->RemainingContent);

  ASSERT_TRUE(isa<InvalidQuery>(Q));
  EXPECT_EQ("1:1: Invalid token <NewLine> found when looking for a value.",
            cast<InvalidQuery>(Q)->ErrStr);

  Q = parse(R"matcher(

let Construct parmVarDecl()

m parmVarDecl(
    Construct
)
)matcher");

  ASSERT_TRUE(isa<LetQuery>(Q));
  {
    llvm::raw_null_ostream NullOutStream;
    dyn_cast<LetQuery>(Q)->run(NullOutStream, QS);
  }

  Q = parse(Q->RemainingContent);

  ASSERT_TRUE(isa<MatchQuery>(Q));
}
