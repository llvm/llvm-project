//===- unittest/ASTMatchers/Dynamic/ParserTest.cpp - Parser unit tests -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-------------------------------------------------------------------===//

#include "../ASTMatchersTest.h"
#include "clang/ASTMatchers/Dynamic/Parser.h"
#include "clang/ASTMatchers/Dynamic/Registry.h"
#include "llvm/ADT/Optional.h"
#include "gtest/gtest.h"
#include <optional>
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
    ExpectedMatchers.insert(std::make_pair(std::string(MatcherName), M));
    return M.getID().second;
  }

  bool isBuilderMatcher(MatcherCtor) const override { return false; }

  ASTNodeKind nodeMatcherType(MatcherCtor) const override { return {}; }

  internal::MatcherDescriptorPtr
  buildMatcherCtor(MatcherCtor, SourceRange NameRange,
                   ArrayRef<ParserValue> Args,
                   Diagnostics *Error) const override {
    return internal::MatcherDescriptorPtr{nullptr};
  }

  void parse(StringRef Code) {
    Diagnostics Error;
    VariantValue Value;
    Parser::parseExpression(Code, this, &Value, &Error);
    Values.push_back(Value);
    Errors.push_back(Error.toStringFull());
  }

  std::optional<MatcherCtor> lookupMatcherCtor(StringRef MatcherName) override {
    const ExpectedMatchersTy::value_type *Matcher =
        &*ExpectedMatchers.find(std::string(MatcherName));
    return reinterpret_cast<MatcherCtor>(Matcher);
  }

  VariantMatcher actOnMatcherExpression(MatcherCtor Ctor,
                                        SourceRange NameRange,
                                        StringRef BindID,
                                        ArrayRef<ParserValue> Args,
                                        Diagnostics *Error) override {
    const ExpectedMatchersTy::value_type *Matcher =
        reinterpret_cast<const ExpectedMatchersTy::value_type *>(Ctor);
    MatcherInfo ToStore = {Matcher->first, NameRange, Args,
                           std::string(BindID)};
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

std::optional<DynTypedMatcher> getSingleMatcher(const VariantValue &Value) {
  std::optional<DynTypedMatcher> Result = Value.getMatcher().getSingleMatcher();
  EXPECT_TRUE(Result);
  return Result;
}

TEST(ParserTest, ParseMatcher) {
  MockSema Sema;
  const uint64_t ExpectedFoo = Sema.expectMatcher("Foo");
  const uint64_t ExpectedBar = Sema.expectMatcher("Bar");
  const uint64_t ExpectedBaz = Sema.expectMatcher("Baz");
  Sema.parse(" Foo ( Bar ( 17), Baz( \n \"B A,Z\") ) .bind( \"Yo!\") ");
  for (const auto &E : Sema.Errors) {
    EXPECT_EQ("", E);
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

TEST(ParserTest, ParseComment) {
  MockSema Sema;
  Sema.expectMatcher("Foo");
  Sema.parse(" Foo() # Bar() ");
  for (const auto &E : Sema.Errors) {
    EXPECT_EQ("", E);
  }

  EXPECT_EQ(1ULL, Sema.Matchers.size());

  Sema.parse("Foo(#) ");

  EXPECT_EQ("1:4: Error parsing matcher. Found end-of-code while looking for ')'.", Sema.Errors[1]);
}

using ast_matchers::internal::Matcher;

Parser::NamedValueMap getTestNamedValues() {
  Parser::NamedValueMap Values;
  Values["nameX"] = llvm::StringRef("x");
  Values["hasParamA"] = VariantMatcher::SingleMatcher(
      functionDecl(hasParameter(0, hasName("a"))));
  return Values;
}

TEST(ParserTest, FullParserTest) {
  Diagnostics Error;

  StringRef Code =
      "varDecl(hasInitializer(binaryOperator(hasLHS(integerLiteral()),"
      "                                      hasOperatorName(\"+\"))))";
  std::optional<DynTypedMatcher> VarDecl(
      Parser::parseMatcherExpression(Code, &Error));
  EXPECT_EQ("", Error.toStringFull());
  Matcher<Decl> M = VarDecl->unconditionalConvertTo<Decl>();
  EXPECT_TRUE(matches("int x = 1 + false;", M));
  EXPECT_FALSE(matches("int x = true + 1;", M));
  EXPECT_FALSE(matches("int x = 1 - false;", M));
  EXPECT_FALSE(matches("int x = true - 1;", M));

  Code = "implicitCastExpr(hasCastKind(\"CK_IntegralToBoolean\"))";
  std::optional<DynTypedMatcher> implicitIntBooleanCast(
      Parser::parseMatcherExpression(Code, nullptr, nullptr, &Error));
  EXPECT_EQ("", Error.toStringFull());
  Matcher<Stmt> MCastStmt =
      traverse(TK_AsIs, implicitIntBooleanCast->unconditionalConvertTo<Stmt>());
  EXPECT_TRUE(matches("bool X = 1;", MCastStmt));
  EXPECT_FALSE(matches("bool X = true;", MCastStmt));

  Code = "functionDecl(hasParameter(1, hasName(\"x\")))";
  std::optional<DynTypedMatcher> HasParameter(
      Parser::parseMatcherExpression(Code, &Error));
  EXPECT_EQ("", Error.toStringFull());
  M = HasParameter->unconditionalConvertTo<Decl>();

  EXPECT_TRUE(matches("void f(int a, int x);", M));
  EXPECT_FALSE(matches("void f(int x, int a);", M));

  // Test named values.
  auto NamedValues = getTestNamedValues();

  Code = "functionDecl(hasParamA, hasParameter(1, hasName(nameX)))";
  std::optional<DynTypedMatcher> HasParameterWithNamedValues(
      Parser::parseMatcherExpression(Code, nullptr, &NamedValues, &Error));
  EXPECT_EQ("", Error.toStringFull());
  M = HasParameterWithNamedValues->unconditionalConvertTo<Decl>();

  EXPECT_TRUE(matches("void f(int a, int x);", M));
  EXPECT_FALSE(matches("void f(int x, int a);", M));

  Code = "unaryExprOrTypeTraitExpr(ofKind(\"UETT_SizeOf\"))";
  std::optional<DynTypedMatcher> UnaryExprSizeOf(
      Parser::parseMatcherExpression(Code, nullptr, nullptr, &Error));
  EXPECT_EQ("", Error.toStringFull());
  Matcher<Stmt> MStmt = UnaryExprSizeOf->unconditionalConvertTo<Stmt>();
  EXPECT_TRUE(matches("unsigned X = sizeof(int);", MStmt));
  EXPECT_FALSE(matches("unsigned X = alignof(int);", MStmt));

  Code =
      R"query(namedDecl(matchesName("^::[ABC]*$", "IgnoreCase | BasicRegex")))query";
  std::optional<DynTypedMatcher> MatchesName(
      Parser::parseMatcherExpression(Code, nullptr, nullptr, &Error));
  EXPECT_EQ("", Error.toStringFull());
  M = MatchesName->unconditionalConvertTo<Decl>();
  EXPECT_TRUE(matches("unsigned AAACCBB;", M));
  EXPECT_TRUE(matches("unsigned aaaccbb;", M));

  Code = "hasInitializer(\n    binaryOperator(hasLHS(\"A\")))";
  EXPECT_TRUE(!Parser::parseMatcherExpression(Code, &Error));
  EXPECT_EQ("1:1: Error parsing argument 1 for matcher hasInitializer.\n"
            "2:5: Error parsing argument 1 for matcher binaryOperator.\n"
            "2:20: Error building matcher hasLHS.\n"
            "2:27: Incorrect type for arg 1. "
            "(Expected = Matcher<Expr>) != (Actual = String)",
            Error.toStringFull());
}

TEST(ParserTest, VariadicMatchTest) {
  Diagnostics Error;

  StringRef Code =
      "stmt(objcMessageExpr(hasAnySelector(\"methodA\", \"methodB:\")))";
  std::optional<DynTypedMatcher> OM(
      Parser::parseMatcherExpression(Code, &Error));
  EXPECT_EQ("", Error.toStringFull());
  auto M = OM->unconditionalConvertTo<Stmt>();
  EXPECT_TRUE(matchesObjC("@interface I @end "
                          "void foo(I* i) { [i methodA]; }", M));
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
  EXPECT_EQ("1:11: Period not followed by valid chained call.",
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
  EXPECT_EQ("1:1: Error building matcher isArrow.\n"
            "1:11: Matcher does not support with call.",
            ParseWithError("isArrow().with"));
  EXPECT_EQ(
      "1:22: Error parsing matcher. Found token <EOF> while looking for '('.",
      ParseWithError("mapAnyOf(ifStmt).with"));
  EXPECT_EQ(
      "1:22: Error parsing matcher. Found end-of-code while looking for ')'.",
      ParseWithError("mapAnyOf(ifStmt).with("));
  EXPECT_EQ("1:1: Failed to build matcher: mapAnyOf.",
            ParseWithError("mapAnyOf()"));
  EXPECT_EQ("1:1: Error parsing argument 1 for matcher mapAnyOf.\n1:1: Failed "
            "to build matcher: mapAnyOf.",
            ParseWithError("mapAnyOf(\"foo\")"));
  EXPECT_EQ("Input value has unresolved overloaded type: "
            "Matcher<DoStmt|ForStmt|WhileStmt|CXXForRangeStmt|FunctionDecl>",
            ParseMatcherWithError("hasBody(stmt())"));
  EXPECT_EQ(
      "1:1: Error parsing argument 1 for matcher decl.\n"
      "1:6: Error building matcher hasAttr.\n"
      "1:14: Unknown value 'attr::Fnal' for arg 1; did you mean 'attr::Final'",
      ParseMatcherWithError(R"query(decl(hasAttr("attr::Fnal")))query"));
  EXPECT_EQ("1:1: Error parsing argument 1 for matcher decl.\n"
            "1:6: Error building matcher hasAttr.\n"
            "1:14: Unknown value 'Final' for arg 1; did you mean 'attr::Final'",
            ParseMatcherWithError(R"query(decl(hasAttr("Final")))query"));
  EXPECT_EQ("1:1: Error parsing argument 1 for matcher decl.\n"
            "1:6: Error building matcher hasAttr.\n"
            "1:14: Value not found: unrelated",
            ParseMatcherWithError(R"query(decl(hasAttr("unrelated")))query"));
  EXPECT_EQ(
      "1:1: Error parsing argument 1 for matcher namedDecl.\n"
      "1:11: Error building matcher matchesName.\n"
      "1:33: Unknown value 'Ignorecase' for arg 2; did you mean 'IgnoreCase'",
      ParseMatcherWithError(
          R"query(namedDecl(matchesName("[ABC]*", "Ignorecase")))query"));
  EXPECT_EQ(
      "1:1: Error parsing argument 1 for matcher namedDecl.\n"
      "1:11: Error building matcher matchesName.\n"
      "1:33: Value not found: IgnoreCase & BasicRegex",
      ParseMatcherWithError(
          R"query(namedDecl(matchesName("[ABC]*", "IgnoreCase & BasicRegex")))query"));
  EXPECT_EQ(
      "1:1: Error parsing argument 1 for matcher namedDecl.\n"
      "1:11: Error building matcher matchesName.\n"
      "1:33: Unknown value 'IgnoreCase | Basicregex' for arg 2; did you mean "
      "'IgnoreCase | BasicRegex'",
      ParseMatcherWithError(
          R"query(namedDecl(matchesName("[ABC]*", "IgnoreCase | Basicregex")))query"));
}

TEST(ParserTest, OverloadErrors) {
  EXPECT_EQ("1:1: Error building matcher callee.\n"
            "1:8: Candidate 1: Incorrect type for arg 1. "
            "(Expected = Matcher<Stmt>) != (Actual = String)\n"
            "1:8: Candidate 2: Incorrect type for arg 1. "
            "(Expected = Matcher<Decl>) != (Actual = String)",
            ParseWithError("callee(\"A\")"));
}

TEST(ParserTest, ParseMultiline) {
  StringRef Code;

  std::optional<DynTypedMatcher> M;
  {
    Code = R"matcher(varDecl(
  hasName("foo")
  )
)matcher";
    Diagnostics Error;
    EXPECT_TRUE(Parser::parseMatcherExpression(Code, &Error));
  }

  {
    Code = R"matcher(varDecl(
  # Internal comment
  hasName("foo") # Internal comment
# Internal comment
  )
)matcher";
    Diagnostics Error;
    EXPECT_TRUE(Parser::parseMatcherExpression(Code, &Error));
  }

  {
    Code = R"matcher(decl().bind(
  "paramName")
)matcher";
    Diagnostics Error;
    EXPECT_TRUE(Parser::parseMatcherExpression(Code, &Error));
  }

  {
    Code = R"matcher(decl().bind(
  "paramName"
  )
)matcher";
    Diagnostics Error;
    EXPECT_TRUE(Parser::parseMatcherExpression(Code, &Error).has_value());
  }

  {
    Code = R"matcher(decl(decl()
, decl()))matcher";
    Diagnostics Error;
    EXPECT_TRUE(Parser::parseMatcherExpression(Code, &Error).has_value());
  }

  {
    Code = R"matcher(decl(decl(),
decl()))matcher";
    Diagnostics Error;
    EXPECT_TRUE(Parser::parseMatcherExpression(Code, &Error).has_value());
  }

  {
    Code = "namedDecl(hasName(\"n\"\n))";
    Diagnostics Error;
    EXPECT_TRUE(Parser::parseMatcherExpression(Code, &Error).has_value());
  }

  {
    Diagnostics Error;

    auto NamedValues = getTestNamedValues();

    Code = R"matcher(hasParamA.bind
  ("paramName")
)matcher";
    M = Parser::parseMatcherExpression(Code, nullptr, &NamedValues, &Error);
    EXPECT_FALSE(M);
    EXPECT_EQ("1:15: Malformed bind() expression.", Error.toStringFull());
  }

  {
    Diagnostics Error;

    auto NamedValues = getTestNamedValues();

    Code = R"matcher(hasParamA.
  bind("paramName")
)matcher";
    M = Parser::parseMatcherExpression(Code, nullptr, &NamedValues, &Error);
    EXPECT_FALSE(M);
    EXPECT_EQ("1:11: Period not followed by valid chained call.",
              Error.toStringFull());
  }

  {
    Diagnostics Error;

    Code = R"matcher(varDecl
()
)matcher";
    M = Parser::parseMatcherExpression(Code, nullptr, nullptr, &Error);
    EXPECT_FALSE(M);
    EXPECT_EQ("1:8: Error parsing matcher. Found token "
              "<NewLine> while looking for '('.",
              Error.toStringFull());
  }

  // Correct line/column numbers
  {
    Diagnostics Error;

    Code = R"matcher(varDecl(
  doesNotExist()
  )
)matcher";
    M = Parser::parseMatcherExpression(Code, nullptr, nullptr, &Error);
    EXPECT_FALSE(M);
    StringRef Expected = R"error(1:1: Error parsing argument 1 for matcher varDecl.
2:3: Matcher not found: doesNotExist)error";
    EXPECT_EQ(Expected, Error.toStringFull());
  }
}

TEST(ParserTest, CompletionRegistry) {
  StringRef Code = "while";
  std::vector<MatcherCompletion> Comps = Parser::completeExpression(Code, 5);
  ASSERT_EQ(1u, Comps.size());
  EXPECT_EQ("Stmt(", Comps[0].TypedText);
  EXPECT_EQ("Matcher<Stmt> whileStmt(Matcher<WhileStmt>...)",
            Comps[0].MatcherDecl);

  Code = "whileStmt().";
  Comps = Parser::completeExpression(Code, 12);
  ASSERT_EQ(1u, Comps.size());
  EXPECT_EQ("bind(\"", Comps[0].TypedText);
  EXPECT_EQ("bind", Comps[0].MatcherDecl);

  Code = "mapAny";
  Comps = Parser::completeExpression(Code, 6);
  ASSERT_EQ(1u, Comps.size());
  EXPECT_EQ("Of(", Comps[0].TypedText);
  EXPECT_EQ("Matcher<NestedNameSpecifierLoc|QualType|TypeLoc|...> "
            "mapAnyOf(NestedNameSpecifierLoc|QualType|TypeLoc|"
            "NestedNameSpecifier|Decl|Stmt|Type...)",
            Comps[0].MatcherDecl);

  Code = "mapAnyOf(ifStmt).";
  Comps = Parser::completeExpression(Code, 17);
  ASSERT_EQ(2u, Comps.size());
  EXPECT_EQ("bind(\"", Comps[0].TypedText);
  EXPECT_EQ("bind", Comps[0].MatcherDecl);
  EXPECT_EQ("with(", Comps[1].TypedText);
  EXPECT_EQ("with", Comps[1].MatcherDecl);

  Code = "mapAnyOf(ifS";
  Comps = Parser::completeExpression(Code, 12);
  ASSERT_EQ(1u, Comps.size());
  EXPECT_EQ("tmt", Comps[0].TypedText);
  EXPECT_EQ("ifStmt", Comps[0].MatcherDecl);
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
  Code = "functionDecl(hasP";
  Comps = Parser::completeExpression(Code, Code.size(), nullptr, &NamedValues);
  ASSERT_EQ(3u, Comps.size());

  EXPECT_EQ("arameter(", Comps[0].TypedText);
  EXPECT_EQ(
      "Matcher<FunctionDecl> hasParameter(unsigned, Matcher<ParmVarDecl>)",
      Comps[0].MatcherDecl);

  EXPECT_EQ("aramA", Comps[1].TypedText);
  EXPECT_EQ("Matcher<Decl> hasParamA", Comps[1].MatcherDecl);

  EXPECT_EQ("arent(", Comps[2].TypedText);
  EXPECT_EQ(
      "Matcher<Decl> "
      "hasParent(Matcher<NestedNameSpecifierLoc|TypeLoc|Decl|...>)",
      Comps[2].MatcherDecl);
}

TEST(ParserTest, ParseBindOnLet) {

  auto NamedValues = getTestNamedValues();

  Diagnostics Error;

  {
    StringRef Code = "hasParamA.bind(\"parmABinding\")";
    std::optional<DynTypedMatcher> TopLevelLetBinding(
        Parser::parseMatcherExpression(Code, nullptr, &NamedValues, &Error));
    EXPECT_EQ("", Error.toStringFull());
    auto M = TopLevelLetBinding->unconditionalConvertTo<Decl>();

    EXPECT_TRUE(matchAndVerifyResultTrue(
        "void foo(int a);", M,
        std::make_unique<VerifyIdIsBoundTo<FunctionDecl>>("parmABinding")));
    EXPECT_TRUE(matchAndVerifyResultFalse(
        "void foo(int b);", M,
        std::make_unique<VerifyIdIsBoundTo<FunctionDecl>>("parmABinding")));
  }

  {
    StringRef Code = "functionDecl(hasParamA.bind(\"parmABinding\"))";
    std::optional<DynTypedMatcher> NestedLetBinding(
        Parser::parseMatcherExpression(Code, nullptr, &NamedValues, &Error));
    EXPECT_EQ("", Error.toStringFull());
    auto M = NestedLetBinding->unconditionalConvertTo<Decl>();

    EXPECT_TRUE(matchAndVerifyResultTrue(
        "void foo(int a);", M,
        std::make_unique<VerifyIdIsBoundTo<FunctionDecl>>("parmABinding")));
    EXPECT_TRUE(matchAndVerifyResultFalse(
        "void foo(int b);", M,
        std::make_unique<VerifyIdIsBoundTo<FunctionDecl>>("parmABinding")));
  }
}

}  // end anonymous namespace
}  // end namespace dynamic
}  // end namespace ast_matchers
}  // end namespace clang
