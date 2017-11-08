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

  EXPECT_EQ(
      str,
      sp(obj{
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

} // namespace
} // namespace json
} // namespace clangd
} // namespace clang
