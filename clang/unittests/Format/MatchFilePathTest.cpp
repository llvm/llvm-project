//===- unittest/Format/MatchFilePathTest.cpp ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../../lib/Format/MatchFilePath.h"
#include "gtest/gtest.h"

namespace clang {
namespace format {
namespace {

class MatchFilePathTest : public testing::Test {
protected:
  bool match(llvm::StringRef FilePath, llvm::StringRef Pattern) {
    return matchFilePath(Pattern, FilePath);
  }
};

// Most of the test cases below are from:
// https://github.com/python/cpython/blob/main/Lib/test/test_fnmatch.py

TEST_F(MatchFilePathTest, Wildcard) {
  EXPECT_TRUE(match("abc", "?*?"));
  EXPECT_TRUE(match("abc", "???*"));
  EXPECT_TRUE(match("abc", "*???"));
  EXPECT_TRUE(match("abc", "???"));
  EXPECT_TRUE(match("abc", "*"));
  EXPECT_TRUE(match("abc", "ab[cd]"));
  EXPECT_TRUE(match("abc", "ab[!de]"));
  EXPECT_FALSE(match("abc", "ab[de]"));
  EXPECT_FALSE(match("a", "??"));
  EXPECT_FALSE(match("a", "b"));
}

TEST_F(MatchFilePathTest, Backslash) {
  EXPECT_TRUE(match("a?", R"(a\?)"));
  EXPECT_FALSE(match("a\\", R"(a\)"));
  EXPECT_TRUE(match("\\", R"([\])"));
  EXPECT_TRUE(match("a", R"([!\])"));
  EXPECT_FALSE(match("\\", R"([!\])"));
}

TEST_F(MatchFilePathTest, Newline) {
  EXPECT_TRUE(match("foo\nbar", "foo*"));
  EXPECT_TRUE(match("foo\nbar\n", "foo*"));
  EXPECT_FALSE(match("\nfoo", "foo*"));
  EXPECT_TRUE(match("\n", "*"));
}

TEST_F(MatchFilePathTest, Star) {
  EXPECT_TRUE(match(std::string(50, 'a'), "*a*a*a*a*a*a*a*a*a*a"));
  EXPECT_FALSE(match(std::string(50, 'a') + 'b', "*a*a*a*a*a*a*a*a*a*a"));
}

TEST_F(MatchFilePathTest, CaseSensitive) {
  EXPECT_TRUE(match("abc", "abc"));
  EXPECT_FALSE(match("AbC", "abc"));
  EXPECT_FALSE(match("abc", "AbC"));
  EXPECT_TRUE(match("AbC", "AbC"));
}

TEST_F(MatchFilePathTest, PathSeparators) {
  EXPECT_TRUE(match("usr/bin", "usr/bin"));
  EXPECT_TRUE(match("usr\\bin", R"(usr\\bin)"));
}

TEST_F(MatchFilePathTest, NumericEscapeSequence) {
  EXPECT_TRUE(match("test", "te*"));
  EXPECT_TRUE(match("test\xff", "te*\xff"));
  EXPECT_TRUE(match("foo\nbar", "foo*"));
}

TEST_F(MatchFilePathTest, ValidBrackets) {
  EXPECT_TRUE(match("z", "[az]"));
  EXPECT_FALSE(match("z", "[!az]"));
  EXPECT_TRUE(match("a", "[aa]"));
  EXPECT_TRUE(match("^", "[^az]"));
  EXPECT_TRUE(match("[", "[[az]"));
  EXPECT_FALSE(match("]", "[!]]"));
}

TEST_F(MatchFilePathTest, InvalidBrackets) {
  EXPECT_TRUE(match("[", "["));
  EXPECT_TRUE(match("[]", "[]"));
  EXPECT_TRUE(match("[!", "[!"));
  EXPECT_TRUE(match("[!]", "[!]"));
}

TEST_F(MatchFilePathTest, Range) {
  EXPECT_TRUE(match("c", "[b-d]"));
  EXPECT_FALSE(match("c", "[!b-d]"));
  EXPECT_TRUE(match("y", "[b-dx-z]"));
  EXPECT_FALSE(match("y", "[!b-dx-z]"));
}

TEST_F(MatchFilePathTest, Hyphen) {
  EXPECT_FALSE(match("#", "[!-#]"));
  EXPECT_FALSE(match("-", "[!--.]"));
  EXPECT_TRUE(match("_", "[^-`]"));
  EXPECT_TRUE(match("]", "[[-^]"));
  EXPECT_TRUE(match("]", R"([\-^])"));
  EXPECT_TRUE(match("-", "[b-]"));
  EXPECT_FALSE(match("-", "[!b-]"));
  EXPECT_TRUE(match("-", "[-b]"));
  EXPECT_FALSE(match("-", "[!-b]"));
  EXPECT_TRUE(match("-", "[-]"));
  EXPECT_FALSE(match("-", "[!-]"));
}

TEST_F(MatchFilePathTest, UpperLELower) {
  EXPECT_FALSE(match("c", "[d-b]"));
  EXPECT_TRUE(match("c", "[!d-b]"));
  EXPECT_TRUE(match("y", "[d-bx-z]"));
  EXPECT_FALSE(match("y", "[!d-bx-z]"));
  EXPECT_TRUE(match("_", "[d-b^-`]"));
  EXPECT_TRUE(match("]", "[d-b[-^]"));
  EXPECT_TRUE(match("b", "[b-b]"));
}

TEST_F(MatchFilePathTest, SlashAndBackslashInBrackets) {
  EXPECT_FALSE(match("/", "[/]"));
  EXPECT_TRUE(match("\\", R"([\])"));
  EXPECT_TRUE(match("[/]", "[/]"));
  EXPECT_TRUE(match("\\", R"([\t])"));
  EXPECT_TRUE(match("t", R"([\t])"));
  EXPECT_FALSE(match("\t", R"([\t])"));
}

TEST_F(MatchFilePathTest, SlashAndBackslashInRange) {
  EXPECT_FALSE(match("a/b", "a[.-0]b"));
  EXPECT_TRUE(match("a\\b", "a[Z-^]b"));
  EXPECT_FALSE(match("a/b", "a[/-0]b"));
  EXPECT_TRUE(match("a[/-0]b", "a[/-0]b"));
  EXPECT_FALSE(match("a/b", "a[.-/]b"));
  EXPECT_TRUE(match("a[.-/]b", "a[.-/]b"));
  EXPECT_TRUE(match("a\\b", R"(a[\-^]b)"));
  EXPECT_TRUE(match("a\\b", R"(a[Z-\]b)"));
}

TEST_F(MatchFilePathTest, Brackets) {
  EXPECT_TRUE(match("[", "[[]"));
  EXPECT_TRUE(match("&", "[a&&b]"));
  EXPECT_TRUE(match("|", "[a||b]"));
  EXPECT_TRUE(match("~", "[a~~b]"));
  EXPECT_TRUE(match(",", "[a-z+--A-Z]"));
  EXPECT_FALSE(match(".", "[a-z--/A-Z]"));
}

TEST_F(MatchFilePathTest, Path) {
  EXPECT_TRUE(match(".clang-format", "*"));
  EXPECT_TRUE(match(".git", "*git*"));
  EXPECT_TRUE(match(".gitignore", "*git*"));
  EXPECT_TRUE(match("foo/bar", "foo*/*bar"));
  EXPECT_TRUE(match("foo/bar", "*/*"));
  EXPECT_TRUE(match("foo/bar", R"(*foo*\/*bar*)"));
  EXPECT_FALSE(match("foo/bar", "foo*"));
  EXPECT_FALSE(match("foo/bar", "foo?bar"));
  EXPECT_FALSE(match("foo/bar", "foo*bar"));
  EXPECT_FALSE(match("foobar", "foo*/*"));
  EXPECT_FALSE(match("foo\\", R"(foo*\)"));
}

} // namespace
} // namespace format
} // namespace clang
