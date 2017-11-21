//===-- JSONExprTests.cpp - JSON expression unit tests ----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "JSONExpr.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace clang {
namespace clangd {
namespace json {
void PrintTo(const Expr &E, std::ostream *OS) {
  llvm::raw_os_ostream(*OS) << llvm::formatv("{0:2}", E);
}
namespace {

std::string s(const Expr &E) { return llvm::formatv("{0}", E).str(); }
std::string sp(const Expr &E) { return llvm::formatv("{0:2}", E).str(); }

TEST(JSONExprTests, Types) {
  EXPECT_EQ("true", s(true));
  EXPECT_EQ("null", s(nullptr));
  EXPECT_EQ("2.5", s(2.5));
  EXPECT_EQ(R"("foo")", s("foo"));
  EXPECT_EQ("[1,2,3]", s({1, 2, 3}));
  EXPECT_EQ(R"({"x":10,"y":20})", s(obj{{"x", 10}, {"y", 20}}));
}

TEST(JSONExprTests, Constructors) {
  // Lots of edge cases around empty and singleton init lists.
  EXPECT_EQ("[[[3]]]", s({{{3}}}));
  EXPECT_EQ("[[[]]]", s({{{}}}));
  EXPECT_EQ("[[{}]]", s({{obj{}}}));
  EXPECT_EQ(R"({"A":{"B":{}}})", s(obj{{"A", obj{{"B", obj{}}}}}));
  EXPECT_EQ(R"({"A":{"B":{"X":"Y"}}})",
            s(obj{{"A", obj{{"B", obj{{"X", "Y"}}}}}}));
}

TEST(JSONExprTests, StringOwnership) {
  char X[] = "Hello";
  Expr Alias = static_cast<const char *>(X);
  X[1] = 'a';
  EXPECT_EQ(R"("Hallo")", s(Alias));

  std::string Y = "Hello";
  Expr Copy = Y;
  Y[1] = 'a';
  EXPECT_EQ(R"("Hello")", s(Copy));
}

TEST(JSONExprTests, CanonicalOutput) {
  // Objects are sorted (but arrays aren't)!
  EXPECT_EQ(R"({"a":1,"b":2,"c":3})", s(obj{{"a", 1}, {"c", 3}, {"b", 2}}));
  EXPECT_EQ(R"(["a","c","b"])", s({"a", "c", "b"}));
  EXPECT_EQ("3", s(3.0));
}

TEST(JSONExprTests, Escaping) {
  std::string test = {
      0,                    // Strings may contain nulls.
      '\b',   '\f',         // Have mnemonics, but we escape numerically.
      '\r',   '\n',   '\t', // Escaped with mnemonics.
      'S',    '\"',   '\\', // Printable ASCII characters.
      '\x7f',               // Delete is not escaped.
      '\xce', '\x94',       // Non-ASCII UTF-8 is not escaped.
  };

  std::string teststring = R"("\u0000\u0008\u000c\r\n\tS\"\\)"
                           "\x7f\xCE\x94\"";

  EXPECT_EQ(teststring, s(test));

  EXPECT_EQ(R"({"object keys are\nescaped":true})",
            s(obj{{"object keys are\nescaped", true}}));
}

TEST(JSONExprTests, PrettyPrinting) {
  const char str[] = R"({
  "empty_array": [],
  "empty_object": {},
  "full_array": [
    1,
    null
  ],
  "full_object": {
    "nested_array": [
      {
        "property": "value"
      }
    ]
  }
})";

  EXPECT_EQ(str, sp(obj{
                     {"empty_object", obj{}},
                     {"empty_array", {}},
                     {"full_array", {1, nullptr}},
                     {"full_object",
                      obj{
                          {"nested_array",
                           {obj{
                               {"property", "value"},
                           }}},
                      }},
                 }));
}

TEST(JSONTest, Parse) {
  auto Compare = [](llvm::StringRef S, Expr Expected) {
    if (auto E = parse(S)) {
      // Compare both string forms and with operator==, in case we have bugs.
      EXPECT_EQ(*E, Expected);
      EXPECT_EQ(sp(*E), sp(Expected));
    } else {
      handleAllErrors(E.takeError(), [S](const llvm::ErrorInfoBase &E) {
        FAIL() << "Failed to parse JSON >>> " << S << " <<<: " << E.message();
      });
    }
  };

  Compare(R"(true)", true);
  Compare(R"(false)", false);
  Compare(R"(null)", nullptr);

  Compare(R"(42)", 42);
  Compare(R"(2.5)", 2.5);
  Compare(R"(2e50)", 2e50);
  Compare(R"(1.2e3456789)", 1.0 / 0.0);

  Compare(R"("foo")", "foo");
  Compare(R"("\"\\\b\f\n\r\t")", "\"\\\b\f\n\r\t");
  Compare(R"("\u0000")", llvm::StringRef("\0", 1));
  Compare("\"\x7f\"", "\x7f");
  Compare(R"("\ud801\udc37")", "\U00010437"); // UTF16 surrogate pair escape.
  Compare("\"\xE2\x82\xAC\xF0\x9D\x84\x9E\"", "\u20ac\U0001d11e"); // UTF8
  Compare(
      R"("LoneLeading=\ud801, LoneTrailing=\udc01, LeadingLeadingTrailing=\ud801\ud801\udc37")",
      "LoneLeading=\ufffd, LoneTrailing=\ufffd, "
      "LeadingLeadingTrailing=\ufffd\U00010437"); // Invalid unicode.

  Compare(R"({"":0,"":0})", obj{{"", 0}});
  Compare(R"({"obj":{},"arr":[]})", obj{{"obj", obj{}}, {"arr", {}}});
  Compare(R"({"\n":{"\u0000":[[[[]]]]}})",
          obj{{"\n", obj{
                         {llvm::StringRef("\0", 1), {{{{}}}}},
                     }}});
  Compare("\r[\n\t] ", {});
}

TEST(JSONTest, ParseErrors) {
  auto ExpectErr = [](llvm::StringRef Msg, llvm::StringRef S) {
    if (auto E = parse(S)) {
      // Compare both string forms and with operator==, in case we have bugs.
      FAIL() << "Parsed JSON >>> " << S << " <<< but wanted error: " << Msg;
    } else {
      handleAllErrors(E.takeError(), [S, Msg](const llvm::ErrorInfoBase &E) {
        EXPECT_THAT(E.message(), testing::HasSubstr(Msg)) << S;
      });
    }
  };
  ExpectErr("Unexpected EOF", "");
  ExpectErr("Unexpected EOF", "[");
  ExpectErr("Text after end of document", "[][]");
  ExpectErr("Text after end of document", "[][]");
  ExpectErr("Invalid bareword", "fuzzy");
  ExpectErr("Expected , or ]", "[2?]");
  ExpectErr("Expected object key", "{a:2}");
  ExpectErr("Expected : after object key", R"({"a",2})");
  ExpectErr("Expected , or } after object property", R"({"a":2 "b":3})");
  ExpectErr("Expected JSON value", R"([&%!])");
  ExpectErr("Invalid number", "1e1.0");
  ExpectErr("Unterminated string", R"("abc\"def)");
  ExpectErr("Control character in string", "\"abc\ndef\"");
  ExpectErr("Invalid escape sequence", R"("\030")");
  ExpectErr("Invalid \\u escape sequence", R"("\usuck")");
  ExpectErr("[3:3, byte=19]", R"({
  "valid": 1,
  invalid: 2
})");
}

} // namespace
} // namespace json
} // namespace clangd
} // namespace clang
