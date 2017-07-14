//===- unittest/ASTMatchers/Dynamic/ParserTest.cpp - Parser unit tests -===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===-------------------------------------------------------------------===//

#include "../ASTMatchersTest.h"
#include "clang/ASTMatchers/Dynamic/Parser.h"
#include "clang/ASTMatchers/Dynamic/Registry.h"
#include "llvm/ADT/Optional.h"
#include "gtest/gtest.h"
#include <string>
#include <vector>

namespace clang {
namespace ast_matchers {
namespace dynamic {
namespace {

class MockSema : public Parser::Sema {
public:
  ~MockSema() override {}

  uint64_t expectMatcher(StringRef MatcherName) {
    // Optimizations on the matcher framework make simple matchers like
    // 'stmt()' to be all the same matcher.
    // Use a more complex expression to prevent that.
    ast_matchers::internal::Matcher<Stmt> M = stmt(stmt(), stmt());
    ExpectedMatchers.insert(std::make_pair(MatcherName, M));
    return M.getID().second;
  }

  void parse(StringRef Code) {
    Diagnostics Error;
    VariantValue Value;
    Parser::parseExpression(Code, this, &Value, &Error);
    Values.push_back(Value);
    Errors.push_back(Error.toStringFull());
  }

  llvm::Optional<MatcherCtor>
  lookupMatcherCtor(StringRef MatcherName) override {
    const ExpectedMatchersTy::value_type *Matcher =
        &*ExpectedMatchers.find(MatcherName);
    return reinterpret_cast<MatcherCtor>(Matcher);
  }

  VariantMatcher actOnMatcherExpression(MatcherCtor Ctor,
                                        SourceRange NameRange,
                                        StringRef BindID,
                                        ArrayRef<ParserValue> Args,
                                        Diagnostics *Error) override {
    const ExpectedMatchersTy::value_type *Matcher =
        reinterpret_cast<const ExpectedMatchersTy::value_type *>(Ctor);
    MatcherInfo ToStore = { Matcher->first, NameRange, Args, BindID };
    Matchers.push_back(ToStore);
    return VariantMatcher::SingleMatcher(Matcher->second);
  }

  struct MatcherInfo {
    StringRef MatcherName;
    SourceRange NameRange;
    std::vector<ParserValue> Args;
    std::string BoundID;
  };

  std::vector<std::string> Errors;
  std::vector<VariantValue> Values;
  std::vector<MatcherInfo> Matchers;
  typedef std::map<std::string, ast_matchers::internal::Matcher<Stmt> >
  ExpectedMatchersTy;
  ExpectedMatchersTy ExpectedMatchers;
};

TEST(ParserTest, ParseBoolean) {
  MockSema Sema;
  Sema.parse("true");
  Sema.parse("false");
  EXPECT_EQ(2U, Sema.Values.size());
  EXPECT_TRUE(Sema.Values[0].getBoolean());
  EXPECT_FALSE(Sema.Values[1].getBoolean());
}

TEST(ParserTest, ParseDouble) {
  MockSema Sema;
  Sema.parse("1.0");
  Sema.parse("2.0f");
  Sema.parse("34.56e-78");
  Sema.parse("4.E+6");
  Sema.parse("1");
  EXPECT_EQ(5U, Sema.Values.size());
  EXPECT_EQ(1.0, Sema.Values[0].getDouble());
  EXPECT_EQ("1:1: Error parsing numeric literal: <2.0f>", Sema.Errors[1]);
  EXPECT_EQ(34.56e-78, Sema.Values[2].getDouble());
  EXPECT_EQ(4e+6, Sema.Values[3].getDouble());
  EXPECT_FALSE(Sema.Values[4].isDouble());
}

TEST(ParserTest, ParseUnsigned) {
  MockSema Sema;
  Sema.parse("0");
  Sema.parse("123");
  Sema.parse("0x1f");
  Sema.parse("12345678901");
  Sema.parse("1a1");
  EXPECT_EQ(5U, Sema.Values.size());
  EXPECT_EQ(0U, Sema.Values[0].getUnsigned());
  EXPECT_EQ(123U, Sema.Values[1].getUnsigned());
  EXPECT_EQ(31U, Sema.Values[2].getUnsigned());
  EXPECT_EQ("1:1: Error parsing numeric literal: <12345678901>", Sema.Errors[3]);
  EXPECT_EQ("1:1: Error parsing numeric literal: <1a1>", Sema.Errors[4]);
}

TEST(ParserTest, ParseString) {
  MockSema Sema;
  Sema.parse("\"Foo\"");
  Sema.parse("\"\"");
  Sema.parse("\"Baz");
  EXPECT_EQ(3ULL, Sema.Values.size());
  EXPECT_EQ("Foo", Sema.Values[0].getString());
  EXPECT_EQ("", Sema.Values[1].getString());
  EXPECT_EQ("1:1: Error parsing string token: <\"Baz>", Sema.Errors[2]);
}

bool matchesRange(SourceRange Range, unsigned StartLine,
                  unsigned EndLine, unsigned StartColumn, unsigned EndColumn) {
  EXPECT_EQ(StartLine, Range.Start.Line);
  EXPECT_EQ(EndLine, Range.End.Line);
  EXPECT_EQ(StartColumn, Range.Start.Column);
  EXPECT_EQ(EndColumn, Range.End.Column);
  return Range.Start.Line == StartLine && Range.End.Line == EndLine &&
         Range.Start.Column == StartColumn && Range.End.Column == EndColumn;
}

llvm::Optional<DynTypedMatcher> getSingleMatcher(const VariantValue &Value) {
  llvm::Optional<DynTypedMatcher> Result =
      Value.getMatcher().getSingleMatcher();
  EXPECT_TRUE(Result.hasValue());
  return Result;
}

TEST(ParserTest, ParseMatcher) {
  MockSema Sema;
  const uint64_t ExpectedFoo = Sema.expectMatcher("Foo");
  const uint64_t ExpectedBar = Sema.expectMatcher("Bar");
  const uint64_t ExpectedBaz = Sema.expectMatcher("Baz");
  Sema.parse(" Foo ( Bar ( 17), Baz( \n \"B A,Z\") ) .bind( \"Yo!\") ");
  for (size_t i = 0, e = Sema.Errors.size(); i != e; ++i) {
    EXPECT_EQ("", Sema.Errors[i]);
  }

  EXPECT_NE(ExpectedFoo, ExpectedBar);
  EXPECT_NE(ExpectedFoo, ExpectedBaz);
  EXPECT_NE(ExpectedBar, ExpectedBaz);

  EXPECT_EQ(1ULL, Sema.Values.size());
  EXPECT_EQ(ExpectedFoo, getSingleMatcher(Sema.Values[0])->getID().second);

  EXPECT_EQ(3ULL, Sema.Matchers.size());
  const MockSema::MatcherInfo Bar = Sema.Matchers[0];
  EXPECT_EQ("Bar", Bar.MatcherName);
  EXPECT_TRUE(matchesRange(Bar.NameRange, 1, 1, 8, 17));
  EXPECT_EQ(1ULL, Bar.Args.size());
  EXPECT_EQ(17U, Bar.Args[0].Value.getUnsigned());

  const MockSema::MatcherInfo Baz = Sema.Matchers[1];
  EXPECT_EQ("Baz", Baz.MatcherName);
  EXPECT_TRUE(matchesRange(Baz.NameRange, 1, 2, 19, 10));
  EXPECT_EQ(1ULL, Baz.Args.size());
  EXPECT_EQ("B A,Z", Baz.Args[0].Value.getString());

  const MockSema::MatcherInfo Foo = Sema.Matchers[2];
  EXPECT_EQ("Foo", Foo.MatcherName);
  EXPECT_TRUE(matchesRange(Foo.NameRange, 1, 2, 2, 12));
  EXPECT_EQ(2ULL, Foo.Args.size());
  EXPECT_EQ(ExpectedBar, getSingleMatcher(Foo.Args[0].Value)->getID().second);
  EXPECT_EQ(ExpectedBaz, getSingleMatcher(Foo.Args[1].Value)->getID().second);
  EXPECT_EQ("Yo!", Foo.BoundID);
}

using ast_matchers::internal::Matcher;

Parser::NamedValueMap getTestNamedValues() {
  Parser::NamedValueMap Values;
  Values["nameX"] = llvm::StringRef("x");
  Values["hasParamA"] =
      VariantMatcher::SingleMatcher(hasParameter(0, hasName("a")));
  return Values;
}

TEST(ParserTest, FullParserTest) {
  Diagnostics Error;
  llvm::Optional<DynTypedMatcher> VarDecl(Parser::parseMatcherExpression(
      "varDecl(hasInitializer(binaryOperator(hasLHS(integerLiteral()),"
      "                                      hasOperatorName(\"+\"))))",
      &Error));
  EXPECT_EQ("", Error.toStringFull());
  Matcher<Decl> M = VarDecl->unconditionalConvertTo<Decl>();
  EXPECT_TRUE(matches("int x = 1 + false;", M));
  EXPECT_FALSE(matches("int x = true + 1;", M));
  EXPECT_FALSE(matches("int x = 1 - false;", M));
  EXPECT_FALSE(matches("int x = true - 1;", M));

  llvm::Optional<DynTypedMatcher> HasParameter(Parser::parseMatcherExpression(
      "functionDecl(hasParameter(1, hasName(\"x\")))", &Error));
  EXPECT_EQ("", Error.toStringFull());
  M = HasParameter->unconditionalConvertTo<Decl>();

  EXPECT_TRUE(matches("void f(int a, int x);", M));
  EXPECT_FALSE(matches("void f(int x, int a);", M));

  // Test named values.
  auto NamedValues = getTestNamedValues();
  llvm::Optional<DynTypedMatcher> HasParameterWithNamedValues(
      Parser::parseMatcherExpression(
          "functionDecl(hasParamA, hasParameter(1, hasName(nameX)))",
          nullptr, &NamedValues, &Error));
  EXPECT_EQ("", Error.toStringFull());
  M = HasParameterWithNamedValues->unconditionalConvertTo<Decl>();

  EXPECT_TRUE(matches("void f(int a, int x);", M));
  EXPECT_FALSE(matches("void f(int x, int a);", M));


  EXPECT_TRUE(!Parser::parseMatcherExpression(
                   "hasInitializer(\n    binaryOperator(hasLHS(\"A\")))",
                   &Error).hasValue());
  EXPECT_EQ("1:1: Error parsing argument 1 for matcher hasInitializer.\n"
            "2:5: Error parsing argument 1 for matcher binaryOperator.\n"
            "2:20: Error building matcher hasLHS.\n"
            "2:27: Incorrect type for arg 1. "
            "(Expected = Matcher<Expr>) != (Actual = String)",
            Error.toStringFull());
}

std::string ParseWithError(StringRef Code) {
  Diagnostics Error;
  VariantValue Value;
  Parser::parseExpression(Code, &Value, &Error);
  return Error.toStringFull();
}

std::string ParseMatcherWithError(StringRef Code) {
  Diagnostics Error;
  Parser::parseMatcherExpression(Code, &Error);
  return Error.toStringFull();
}

TEST(ParserTest, Errors) {
  EXPECT_EQ(
      "1:5: Error parsing matcher. Found token <123> while looking for '('.",
      ParseWithError("Foo 123"));
  EXPECT_EQ(
      "1:1: Matcher not found: Foo\n"
      "1:9: Error parsing matcher. Found token <123> while looking for ','.",
      ParseWithError("Foo(\"A\" 123)"));
  EXPECT_EQ(
      "1:1: Error parsing argument 1 for matcher stmt.\n"
      "1:6: Value not found: someValue",
      ParseWithError("stmt(someValue)"));
  EXPECT_EQ(
      "1:1: Matcher not found: Foo\n"
      "1:4: Error parsing matcher. Found end-of-code while looking for ')'.",
      ParseWithError("Foo("));
  EXPECT_EQ("1:1: End of code found while looking for token.",
            ParseWithError(""));
  EXPECT_EQ("Input value is not a matcher expression.",
            ParseMatcherWithError("\"A\""));
  EXPECT_EQ("1:1: Matcher not found: Foo\n"
            "1:1: Error parsing argument 1 for matcher Foo.\n"
            "1:5: Invalid token <(> found when looking for a value.",
            ParseWithError("Foo(("));
  EXPECT_EQ("1:7: Expected end of code.", ParseWithError("expr()a"));
  EXPECT_EQ("1:11: Malformed bind() expression.",
            ParseWithError("isArrow().biind"));
  EXPECT_EQ("1:15: Malformed bind() expression.",
            ParseWithError("isArrow().bind"));
  EXPECT_EQ("1:16: Malformed bind() expression.",
            ParseWithError("isArrow().bind(foo"));
  EXPECT_EQ("1:21: Malformed bind() expression.",
            ParseWithError("isArrow().bind(\"foo\""));
  EXPECT_EQ("1:1: Error building matcher isArrow.\n"
            "1:1: Matcher does not support binding.",
            ParseWithError("isArrow().bind(\"foo\")"));
  EXPECT_EQ("Input value has unresolved overloaded type: "
            "Matcher<DoStmt|ForStmt|WhileStmt|CXXForRangeStmt|FunctionDecl>",
            ParseMatcherWithError("hasBody(stmt())"));
}

TEST(ParserTest, OverloadErrors) {
  EXPECT_EQ("1:1: Error building matcher callee.\n"
            "1:8: Candidate 1: Incorrect type for arg 1. "
            "(Expected = Matcher<Stmt>) != (Actual = String)\n"
            "1:8: Candidate 2: Incorrect type for arg 1. "
            "(Expected = Matcher<Decl>) != (Actual = String)",
            ParseWithError("callee(\"A\")"));
}

TEST(ParserTest, CompletionRegistry) {
  std::vector<MatcherCompletion> Comps =
      Parser::completeExpression("while", 5);
  ASSERT_EQ(1u, Comps.size());
  EXPECT_EQ("Stmt(", Comps[0].TypedText);
  EXPECT_EQ("Matcher<Stmt> whileStmt(Matcher<WhileStmt>...)",
            Comps[0].MatcherDecl);

  Comps = Parser::completeExpression("whileStmt().", 12);
  ASSERT_EQ(1u, Comps.size());
  EXPECT_EQ("bind(\"", Comps[0].TypedText);
  EXPECT_EQ("bind", Comps[0].MatcherDecl);
}

TEST(ParserTest, CompletionNamedValues) {
  // Can complete non-matcher types.
  auto NamedValues = getTestNamedValues();
  StringRef Code = "functionDecl(hasName(";
  std::vector<MatcherCompletion> Comps =
      Parser::completeExpression(Code, Code.size(), nullptr, &NamedValues);
  ASSERT_EQ(1u, Comps.size());
  EXPECT_EQ("nameX", Comps[0].TypedText);
  EXPECT_EQ("String nameX", Comps[0].MatcherDecl);

  // Can complete if there are names in the expression.
  Code = "cxxMethodDecl(hasName(nameX), ";
  Comps = Parser::completeExpression(Code, Code.size(), nullptr, &NamedValues);
  EXPECT_LT(0u, Comps.size());

  // Can complete names and registry together.
  Code = "cxxMethodDecl(hasP";
  Comps = Parser::completeExpression(Code, Code.size(), nullptr, &NamedValues);
  ASSERT_EQ(3u, Comps.size());
  EXPECT_EQ("aramA", Comps[0].TypedText);
  EXPECT_EQ("Matcher<FunctionDecl> hasParamA", Comps[0].MatcherDecl);

  EXPECT_EQ("arameter(", Comps[1].TypedText);
  EXPECT_EQ(
      "Matcher<FunctionDecl> hasParameter(unsigned, Matcher<ParmVarDecl>)",
      Comps[1].MatcherDecl);

  EXPECT_EQ("arent(", Comps[2].TypedText);
  EXPECT_EQ(
      "Matcher<Decl> "
      "hasParent(Matcher<NestedNameSpecifierLoc|TypeLoc|Decl|...>)",
      Comps[2].MatcherDecl);
}

}  // end anonymous namespace
}  // end namespace dynamic
}  // end namespace ast_matchers
}  // end namespace clang
