//===-- FormattedStringTests.cpp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "FormattedString.h"
#include "clang/Basic/LLVM.h"
#include "llvm/ADT/StringRef.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace clang {
namespace clangd {
namespace {

TEST(FormattedString, Basic) {
  FormattedString S;
  EXPECT_EQ(S.renderAsPlainText(), "");
  EXPECT_EQ(S.renderAsMarkdown(), "");

  S.appendText("foobar");
  EXPECT_EQ(S.renderAsPlainText(), "foobar");
  EXPECT_EQ(S.renderAsMarkdown(), "foobar");

  S = FormattedString();
  S.appendInlineCode("foobar");
  EXPECT_EQ(S.renderAsPlainText(), "foobar");
  EXPECT_EQ(S.renderAsMarkdown(), "`foobar`");

  S = FormattedString();
  S.appendCodeBlock("foobar");
  EXPECT_EQ(S.renderAsPlainText(), "foobar");
  EXPECT_EQ(S.renderAsMarkdown(), "```cpp\n"
                                  "foobar\n"
                                  "```\n");
}

TEST(FormattedString, CodeBlocks) {
  FormattedString S;
  S.appendCodeBlock("foobar");
  S.appendCodeBlock("bazqux", "javascript");

  EXPECT_EQ(S.renderAsPlainText(), "foobar\n\n\nbazqux");
  std::string ExpectedMarkdown = R"md(```cpp
foobar
```
```javascript
bazqux
```
)md";
  EXPECT_EQ(S.renderAsMarkdown(), ExpectedMarkdown);

  S = FormattedString();
  S.appendInlineCode("foobar");
  S.appendInlineCode("bazqux");
  EXPECT_EQ(S.renderAsPlainText(), "foobar bazqux");
  EXPECT_EQ(S.renderAsMarkdown(), "`foobar` `bazqux`");

  S = FormattedString();
  S.appendText("foo");
  S.appendInlineCode("bar");
  S.appendText("baz");

  EXPECT_EQ(S.renderAsPlainText(), "foo bar baz");
  EXPECT_EQ(S.renderAsMarkdown(), "foo`bar`baz");
}

TEST(FormattedString, Escaping) {
  // Check some ASCII punctuation
  FormattedString S;
  S.appendText("*!`");
  EXPECT_EQ(S.renderAsMarkdown(), "\\*\\!\\`");

  // Check all ASCII punctuation.
  S = FormattedString();
  std::string Punctuation = R"txt(!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~)txt";
  // Same text, with each character escaped.
  std::string EscapedPunctuation;
  EscapedPunctuation.reserve(2 * Punctuation.size());
  for (char C : Punctuation)
    EscapedPunctuation += std::string("\\") + C;
  S.appendText(Punctuation);
  EXPECT_EQ(S.renderAsMarkdown(), EscapedPunctuation);

  // In code blocks we don't need to escape ASCII punctuation.
  S = FormattedString();
  S.appendInlineCode("* foo !+ bar * baz");
  EXPECT_EQ(S.renderAsMarkdown(), "`* foo !+ bar * baz`");
  S = FormattedString();
  S.appendCodeBlock("#define FOO\n* foo !+ bar * baz");
  EXPECT_EQ(S.renderAsMarkdown(), "```cpp\n"
                                  "#define FOO\n* foo !+ bar * baz\n"
                                  "```\n");

  // But we have to escape the backticks.
  S = FormattedString();
  S.appendInlineCode("foo`bar`baz");
  EXPECT_EQ(S.renderAsMarkdown(), "`foo``bar``baz`");

  S = FormattedString();
  S.appendCodeBlock("foo`bar`baz");
  EXPECT_EQ(S.renderAsMarkdown(), "```cpp\n"
                                  "foo`bar`baz\n"
                                  "```\n");

  // Inline code blocks starting or ending with backticks should add spaces.
  S = FormattedString();
  S.appendInlineCode("`foo");
  EXPECT_EQ(S.renderAsMarkdown(), "` ``foo `");
  S = FormattedString();
  S.appendInlineCode("foo`");
  EXPECT_EQ(S.renderAsMarkdown(), "` foo`` `");
  S = FormattedString();
  S.appendInlineCode("`foo`");
  EXPECT_EQ(S.renderAsMarkdown(), "` ``foo`` `");

  // Should also add extra spaces if the block stars and ends with spaces.
  S = FormattedString();
  S.appendInlineCode(" foo ");
  EXPECT_EQ(S.renderAsMarkdown(), "`  foo  `");
  S = FormattedString();
  S.appendInlineCode("foo ");
  EXPECT_EQ(S.renderAsMarkdown(), "`foo `");
  S = FormattedString();
  S.appendInlineCode(" foo");
  EXPECT_EQ(S.renderAsMarkdown(), "` foo`");

  // Code blocks might need more than 3 backticks.
  S = FormattedString();
  S.appendCodeBlock("foobarbaz `\nqux");
  EXPECT_EQ(S.renderAsMarkdown(), "```cpp\n"
                                  "foobarbaz `\nqux\n"
                                  "```\n");
  S = FormattedString();
  S.appendCodeBlock("foobarbaz ``\nqux");
  EXPECT_EQ(S.renderAsMarkdown(), "```cpp\n"
                                  "foobarbaz ``\nqux\n"
                                  "```\n");
  S = FormattedString();
  S.appendCodeBlock("foobarbaz ```\nqux");
  EXPECT_EQ(S.renderAsMarkdown(), "````cpp\n"
                                  "foobarbaz ```\nqux\n"
                                  "````\n");
  S = FormattedString();
  S.appendCodeBlock("foobarbaz ` `` ``` ```` `\nqux");
  EXPECT_EQ(S.renderAsMarkdown(), "`````cpp\n"
                                  "foobarbaz ` `` ``` ```` `\nqux\n"
                                  "`````\n");
}

} // namespace
} // namespace clangd
} // namespace clang
