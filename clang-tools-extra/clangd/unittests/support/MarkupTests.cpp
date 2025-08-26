//===-- MarkupTests.cpp ---------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "support/Markup.h"
#include "clang/Basic/LLVM.h"
#include "llvm/ADT/StringRef.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace clang {
namespace clangd {
namespace markup {
namespace {

std::string escape(llvm::StringRef Text) {
  return Paragraph().appendText(Text.str()).asEscapedMarkdown();
}

std::string dontEscape(llvm::StringRef Text) {
  return Paragraph().appendText(Text.str()).asMarkdown();
}

MATCHER_P(escaped, C, "") {
  return testing::ExplainMatchResult(::testing::HasSubstr(std::string{'\\', C}),
                                     arg, result_listener);
}

MATCHER(escapedNone, "") {
  return testing::ExplainMatchResult(::testing::Not(::testing::HasSubstr("\\")),
                                     arg, result_listener);
}

TEST(Render, Escaping) {
  // Check all ASCII punctuation.
  std::string Punctuation = R"txt(!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~)txt";
  std::string EscapedPunc = R"txt(!"#$%&'()\*+,-./:;<=>?@[\\]^\_\`{|}~)txt";
  EXPECT_EQ(escape(Punctuation), EscapedPunc);

  // Inline code
  EXPECT_EQ(escape("`foo`"), R"(\`foo\`)");
  EXPECT_EQ(escape("`foo"), R"(\`foo)");
  EXPECT_EQ(escape("foo`"), R"(foo\`)");
  EXPECT_EQ(escape("``foo``"), R"(\`\`foo\`\`)");
  // Code blocks
  EXPECT_EQ(escape("```"), R"(\`\`\`)"); // This could also be inline code!
  EXPECT_EQ(escape("~~~"), R"(\~~~)");

  // Rulers and headings
  EXPECT_THAT(escape("## Heading"), escaped('#'));
  EXPECT_THAT(escape("Foo # bar"), escapedNone());
  EXPECT_EQ(escape("---"), R"(\---)");
  EXPECT_EQ(escape("-"), R"(\-)");
  EXPECT_EQ(escape("==="), R"(\===)");
  EXPECT_EQ(escape("="), R"(\=)");
  EXPECT_EQ(escape("***"), R"(\*\*\*)"); // \** could start emphasis!

  // HTML tags.
  EXPECT_THAT(escape("<pre"), escaped('<'));
  EXPECT_THAT(escape("< pre"), escapedNone());
  EXPECT_THAT(escape("if a<b then"), escaped('<'));
  EXPECT_THAT(escape("if a<b then c."), escapedNone());
  EXPECT_THAT(escape("if a<b then c='foo'."), escaped('<'));
  EXPECT_THAT(escape("std::vector<T>"), escaped('<'));
  EXPECT_THAT(escape("std::vector<std::string>"), escaped('<'));
  EXPECT_THAT(escape("std::map<int, int>"), escapedNone());
  // Autolinks
  EXPECT_THAT(escape("Email <foo@bar.com>"), escapedNone());
  EXPECT_THAT(escape("Website <http://foo.bar>"), escapedNone());

  // Bullet lists.
  EXPECT_THAT(escape("- foo"), escaped('-'));
  EXPECT_THAT(escape("* foo"), escaped('*'));
  EXPECT_THAT(escape("+ foo"), escaped('+'));
  EXPECT_THAT(escape("+"), escaped('+'));
  EXPECT_THAT(escape("a + foo"), escapedNone());
  EXPECT_THAT(escape("a+ foo"), escapedNone());
  EXPECT_THAT(escape("1. foo"), escaped('.'));
  EXPECT_THAT(escape("a. foo"), escapedNone());

  // Emphasis.
  EXPECT_EQ(escape("*foo*"), R"(\*foo\*)");
  EXPECT_EQ(escape("**foo**"), R"(\*\*foo\*\*)");
  EXPECT_THAT(escape("*foo"), escaped('*'));
  EXPECT_THAT(escape("foo *"), escapedNone());
  EXPECT_THAT(escape("foo * bar"), escapedNone());
  EXPECT_THAT(escape("foo_bar"), escapedNone());
  EXPECT_THAT(escape("foo _bar"), escaped('_'));
  EXPECT_THAT(escape("foo_ bar"), escaped('_'));
  EXPECT_THAT(escape("foo _ bar"), escapedNone());

  // HTML entities.
  EXPECT_THAT(escape("fish &chips;"), escaped('&'));
  EXPECT_THAT(escape("fish & chips;"), escapedNone());
  EXPECT_THAT(escape("fish &chips"), escapedNone());
  EXPECT_THAT(escape("foo &#42; bar"), escaped('&'));
  EXPECT_THAT(escape("foo &#xaf; bar"), escaped('&'));
  EXPECT_THAT(escape("foo &?; bar"), escapedNone());

  // Links.
  EXPECT_THAT(escape("[foo](bar)"), escaped(']'));
  EXPECT_THAT(escape("[foo]: bar"), escaped(']'));
  // No need to escape these, as the target never exists.
  EXPECT_THAT(escape("[foo][]"), escapedNone());
  EXPECT_THAT(escape("[foo][bar]"), escapedNone());
  EXPECT_THAT(escape("[foo]"), escapedNone());

  // In code blocks we don't need to escape ASCII punctuation.
  Paragraph P = Paragraph();
  P.appendCode("* foo !+ bar * baz");
  EXPECT_EQ(P.asEscapedMarkdown(), "`* foo !+ bar * baz`");

  // But we have to escape the backticks.
  P = Paragraph();
  P.appendCode("foo`bar`baz", /*Preserve=*/true);
  EXPECT_EQ(P.asEscapedMarkdown(), "`foo``bar``baz`");
  // In plain-text, we fall back to different quotes.
  EXPECT_EQ(P.asPlainText(), "'foo`bar`baz'");

  // Inline code blocks starting or ending with backticks should add spaces.
  P = Paragraph();
  P.appendCode("`foo");
  EXPECT_EQ(P.asEscapedMarkdown(), "` ``foo `");
  P = Paragraph();
  P.appendCode("foo`");
  EXPECT_EQ(P.asEscapedMarkdown(), "` foo`` `");
  P = Paragraph();
  P.appendCode("`foo`");
  EXPECT_EQ(P.asEscapedMarkdown(), "` ``foo`` `");

  // Code blocks might need more than 3 backticks.
  Document D;
  D.addCodeBlock("foobarbaz `\nqux");
  EXPECT_EQ(D.asEscapedMarkdown(), "```cpp\n"
                                   "foobarbaz `\nqux\n"
                                   "```");
  D = Document();
  D.addCodeBlock("foobarbaz ``\nqux");
  EXPECT_THAT(D.asEscapedMarkdown(), "```cpp\n"
                                     "foobarbaz ``\nqux\n"
                                     "```");
  D = Document();
  D.addCodeBlock("foobarbaz ```\nqux");
  EXPECT_EQ(D.asEscapedMarkdown(), "````cpp\n"
                                   "foobarbaz ```\nqux\n"
                                   "````");
  D = Document();
  D.addCodeBlock("foobarbaz ` `` ``` ```` `\nqux");
  EXPECT_EQ(D.asEscapedMarkdown(), "`````cpp\n"
                                   "foobarbaz ` `` ``` ```` `\nqux\n"
                                   "`````");
}

TEST(Render, NoEscaping) {
  // Check all ASCII punctuation.
  std::string Punctuation = R"txt(!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~)txt";
  EXPECT_EQ(dontEscape(Punctuation), Punctuation);

  // Inline code
  EXPECT_THAT(dontEscape("`foo`"), escapedNone());
  EXPECT_THAT(dontEscape("`foo"), escapedNone());
  EXPECT_THAT(dontEscape("foo`"), escapedNone());
  EXPECT_THAT(dontEscape("``foo``"), escapedNone());
  // Code blocks
  EXPECT_THAT(dontEscape("```"), escapedNone());
  EXPECT_THAT(dontEscape("~~~"), escapedNone());

  // Rulers and headings
  EXPECT_THAT(dontEscape("## Heading"), escapedNone());
  EXPECT_THAT(dontEscape("Foo # bar"), escapedNone());
  EXPECT_THAT(dontEscape("---"), escapedNone());
  EXPECT_THAT(dontEscape("-"), escapedNone());
  EXPECT_THAT(dontEscape("==="), escapedNone());
  EXPECT_THAT(dontEscape("="), escapedNone());
  EXPECT_THAT(dontEscape("***"), escapedNone()); // \** could start emphasis!

  // HTML tags.
  EXPECT_THAT(dontEscape("<pre"), escaped('<'));
  EXPECT_THAT(dontEscape("< pre"), escapedNone());
  EXPECT_THAT(dontEscape("if a<b then"), escaped('<'));
  EXPECT_THAT(dontEscape("if a<b then c."), escapedNone());
  EXPECT_THAT(dontEscape("if a<b then c='foo'."), escaped('<'));
  EXPECT_THAT(dontEscape("std::vector<T>"), escaped('<'));
  EXPECT_THAT(dontEscape("std::vector<std::string>"), escaped('<'));
  EXPECT_THAT(dontEscape("std::map<int, int>"), escapedNone());
  // Autolinks
  EXPECT_THAT(dontEscape("Email <foo@bar.com>"), escapedNone());
  EXPECT_THAT(dontEscape("Website <http://foo.bar>"), escapedNone());

  // Bullet lists.
  EXPECT_THAT(dontEscape("- foo"), escapedNone());
  EXPECT_THAT(dontEscape("* foo"), escapedNone());
  EXPECT_THAT(dontEscape("+ foo"), escapedNone());
  EXPECT_THAT(dontEscape("+"), escapedNone());
  EXPECT_THAT(dontEscape("a + foo"), escapedNone());
  EXPECT_THAT(dontEscape("a+ foo"), escapedNone());
  EXPECT_THAT(dontEscape("1. foo"), escapedNone());
  EXPECT_THAT(dontEscape("a. foo"), escapedNone());

  // Emphasis.
  EXPECT_THAT(dontEscape("*foo*"), escapedNone());
  EXPECT_THAT(dontEscape("**foo**"), escapedNone());
  EXPECT_THAT(dontEscape("*foo"), escapedNone());
  EXPECT_THAT(dontEscape("foo *"), escapedNone());
  EXPECT_THAT(dontEscape("foo * bar"), escapedNone());
  EXPECT_THAT(dontEscape("foo_bar"), escapedNone());
  EXPECT_THAT(dontEscape("foo _bar"), escapedNone());
  EXPECT_THAT(dontEscape("foo_ bar"), escapedNone());
  EXPECT_THAT(dontEscape("foo _ bar"), escapedNone());

  // HTML entities.
  EXPECT_THAT(dontEscape("fish &chips;"), escaped('&'));
  EXPECT_THAT(dontEscape("fish & chips;"), escapedNone());
  EXPECT_THAT(dontEscape("fish &chips"), escapedNone());
  EXPECT_THAT(dontEscape("foo &#42; bar"), escaped('&'));
  EXPECT_THAT(dontEscape("foo &#xaf; bar"), escaped('&'));
  EXPECT_THAT(dontEscape("foo &?; bar"), escapedNone());

  // Links.
  EXPECT_THAT(dontEscape("[foo](bar)"), escapedNone());
  EXPECT_THAT(dontEscape("[foo]: bar"), escapedNone());
  // No need to escape these, as the target never exists.
  EXPECT_THAT(dontEscape("[foo][]"), escapedNone());
  EXPECT_THAT(dontEscape("[foo][bar]"), escapedNone());
  EXPECT_THAT(dontEscape("[foo]"), escapedNone());

  // In code blocks we don't need to escape ASCII punctuation.
  Paragraph P = Paragraph();
  P.appendCode("* foo !+ bar * baz");
  EXPECT_EQ(P.asMarkdown(), "`* foo !+ bar * baz`");

  // But we have to escape the backticks.
  P = Paragraph();
  P.appendCode("foo`bar`baz", /*Preserve=*/true);
  EXPECT_EQ(P.asMarkdown(), "`foo``bar``baz`");

  // Inline code blocks starting or ending with backticks should add spaces.
  P = Paragraph();
  P.appendCode("`foo");
  EXPECT_EQ(P.asMarkdown(), "` ``foo `");
  P = Paragraph();
  P.appendCode("foo`");
  EXPECT_EQ(P.asMarkdown(), "` foo`` `");
  P = Paragraph();
  P.appendCode("`foo`");
  EXPECT_EQ(P.asMarkdown(), "` ``foo`` `");

  // Code blocks might need more than 3 backticks.
  Document D;
  D.addCodeBlock("foobarbaz `\nqux");
  EXPECT_EQ(D.asMarkdown(), "```cpp\n"
                            "foobarbaz `\nqux\n"
                            "```");
  D = Document();
  D.addCodeBlock("foobarbaz ``\nqux");
  EXPECT_THAT(D.asMarkdown(), "```cpp\n"
                              "foobarbaz ``\nqux\n"
                              "```");
  D = Document();
  D.addCodeBlock("foobarbaz ```\nqux");
  EXPECT_EQ(D.asMarkdown(), "````cpp\n"
                            "foobarbaz ```\nqux\n"
                            "````");
  D = Document();
  D.addCodeBlock("foobarbaz ` `` ``` ```` `\nqux");
  EXPECT_EQ(D.asMarkdown(), "`````cpp\n"
                            "foobarbaz ` `` ``` ```` `\nqux\n"
                            "`````");
}

TEST(Paragraph, SeparationOfChunks) {
  // This test keeps appending contents to a single Paragraph and checks
  // expected accumulated contents after each one.
  // Purpose is to check for separation between different chunks.
  Paragraph P;

  P.appendText("after ");
  EXPECT_EQ(P.asEscapedMarkdown(), "after");
  EXPECT_EQ(P.asMarkdown(), "after");
  EXPECT_EQ(P.asPlainText(), "after");

  P.appendCode("foobar").appendSpace();
  EXPECT_EQ(P.asEscapedMarkdown(), "after `foobar`");
  EXPECT_EQ(P.asMarkdown(), "after `foobar`");
  EXPECT_EQ(P.asPlainText(), "after foobar");

  P.appendText("bat");
  EXPECT_EQ(P.asEscapedMarkdown(), "after `foobar` bat");
  EXPECT_EQ(P.asMarkdown(), "after `foobar` bat");
  EXPECT_EQ(P.asPlainText(), "after foobar bat");

  P.appendCode("no").appendCode("space");
  EXPECT_EQ(P.asEscapedMarkdown(), "after `foobar` bat`no` `space`");
  EXPECT_EQ(P.asMarkdown(), "after `foobar` bat`no` `space`");
  EXPECT_EQ(P.asPlainText(), "after foobar batno space");

  P.appendText(" text");
  EXPECT_EQ(P.asEscapedMarkdown(), "after `foobar` bat`no` `space` text");
  EXPECT_EQ(P.asMarkdown(), "after `foobar` bat`no` `space` text");
  EXPECT_EQ(P.asPlainText(), "after foobar batno space text");

  P.appendSpace().appendCode("code").appendText(".\n  newline");
  EXPECT_EQ(P.asEscapedMarkdown(),
            "after `foobar` bat`no` `space` text `code`.\n  newline");
  EXPECT_EQ(P.asMarkdown(),
            "after `foobar` bat`no` `space` text `code`.\n  newline");
  EXPECT_EQ(P.asPlainText(), "after foobar batno space text code.\nnewline");
}

TEST(Paragraph, SeparationOfChunks2) {
  // This test keeps appending contents to a single Paragraph and checks
  // expected accumulated contents after each one.
  // Purpose is to check for separation between different chunks
  // where the spacing is in the appended string rather set by appendSpace.
  Paragraph P;

  P.appendText("after ");
  EXPECT_EQ(P.asEscapedMarkdown(), "after");
  EXPECT_EQ(P.asMarkdown(), "after");
  EXPECT_EQ(P.asPlainText(), "after");

  P.appendText("foobar");
  EXPECT_EQ(P.asEscapedMarkdown(), "after foobar");
  EXPECT_EQ(P.asMarkdown(), "after foobar");
  EXPECT_EQ(P.asPlainText(), "after foobar");

  P.appendText(" bat");
  EXPECT_EQ(P.asEscapedMarkdown(), "after foobar bat");
  EXPECT_EQ(P.asMarkdown(), "after foobar bat");
  EXPECT_EQ(P.asPlainText(), "after foobar bat");

  P.appendText("baz");
  EXPECT_EQ(P.asEscapedMarkdown(), "after foobar batbaz");
  EXPECT_EQ(P.asMarkdown(), "after foobar batbaz");
  EXPECT_EQ(P.asPlainText(), "after foobar batbaz");

  P.appendText(" faz ");
  EXPECT_EQ(P.asEscapedMarkdown(), "after foobar batbaz faz");
  EXPECT_EQ(P.asMarkdown(), "after foobar batbaz faz");
  EXPECT_EQ(P.asPlainText(), "after foobar batbaz faz");

  P.appendText("  bar  ");
  EXPECT_EQ(P.asEscapedMarkdown(), "after foobar batbaz faz   bar");
  EXPECT_EQ(P.asMarkdown(), "after foobar batbaz faz   bar");
  EXPECT_EQ(P.asPlainText(), "after foobar batbaz faz bar");

  P.appendText("qux");
  EXPECT_EQ(P.asEscapedMarkdown(), "after foobar batbaz faz   bar  qux");
  EXPECT_EQ(P.asMarkdown(), "after foobar batbaz faz   bar  qux");
  EXPECT_EQ(P.asPlainText(), "after foobar batbaz faz bar qux");
}

TEST(Paragraph, SeparationOfChunks3) {
  // This test keeps appending contents to a single Paragraph and checks
  // expected accumulated contents after each one.
  // Purpose is to check for separation between different chunks
  // where the spacing is in the appended string rather set by appendSpace.
  Paragraph P;

  P.appendText("after  \n");
  EXPECT_EQ(P.asEscapedMarkdown(), "after");
  EXPECT_EQ(P.asMarkdown(), "after");
  EXPECT_EQ(P.asPlainText(), "after");

  P.appendText("  foobar\n");
  EXPECT_EQ(P.asEscapedMarkdown(), "after  \n  foobar");
  EXPECT_EQ(P.asMarkdown(), "after  \n  foobar");
  EXPECT_EQ(P.asPlainText(), "after\nfoobar");

  P.appendText("- bat\n");
  EXPECT_EQ(P.asEscapedMarkdown(), "after  \n  foobar\n\\- bat");
  EXPECT_EQ(P.asMarkdown(), "after  \n  foobar\n- bat");
  EXPECT_EQ(P.asPlainText(), "after\nfoobar\n- bat");

  P.appendText("- baz");
  EXPECT_EQ(P.asEscapedMarkdown(), "after  \n  foobar\n\\- bat\n\\- baz");
  EXPECT_EQ(P.asMarkdown(), "after  \n  foobar\n- bat\n- baz");
  EXPECT_EQ(P.asPlainText(), "after\nfoobar\n- bat\n- baz");

  P.appendText(" faz ");
  EXPECT_EQ(P.asEscapedMarkdown(), "after  \n  foobar\n\\- bat\n\\- baz faz");
  EXPECT_EQ(P.asMarkdown(), "after  \n  foobar\n- bat\n- baz faz");
  EXPECT_EQ(P.asPlainText(), "after\nfoobar\n- bat\n- baz faz");
}

TEST(Paragraph, ExtraSpaces) {
  // Make sure spaces inside chunks are preserved for markdown
  // and dropped for plain text.
  Paragraph P;
  P.appendText("foo\n   \t   baz");
  P.appendCode(" bar\n");
  EXPECT_EQ(P.asEscapedMarkdown(), "foo\n   \t   baz`bar`");
  EXPECT_EQ(P.asMarkdown(), "foo\n   \t   baz`bar`");
  EXPECT_EQ(P.asPlainText(), "foo bazbar");
}

TEST(Paragraph, SpacesCollapsed) {
  Paragraph P;
  P.appendText(" foo bar ");
  P.appendText(" baz ");
  EXPECT_EQ(P.asEscapedMarkdown(), "foo bar  baz");
  EXPECT_EQ(P.asMarkdown(), "foo bar  baz");
  EXPECT_EQ(P.asPlainText(), "foo bar baz");
}

TEST(Paragraph, NewLines) {
  // New lines before and after chunks are dropped.
  Paragraph P;
  P.appendText(" \n foo\nbar\n ");
  P.appendCode(" \n foo\nbar \n ");
  EXPECT_EQ(P.asEscapedMarkdown(), "foo\nbar\n `foo bar`");
  EXPECT_EQ(P.asMarkdown(), "foo\nbar\n `foo bar`");
  EXPECT_EQ(P.asPlainText(), "foo bar foo bar");
}

TEST(Paragraph, BoldText) {
  Paragraph P;
  P.appendBoldText("");
  EXPECT_EQ(P.asEscapedMarkdown(), "");
  EXPECT_EQ(P.asMarkdown(), "");
  EXPECT_EQ(P.asPlainText(), "");

  P.appendBoldText(" \n foo\nbar\n ");
  EXPECT_EQ(P.asEscapedMarkdown(), "\\*\\*foo bar\\*\\*");
  EXPECT_EQ(P.asMarkdown(), "**foo bar**");
  EXPECT_EQ(P.asPlainText(), "**foo bar**");

  P.appendSpace().appendBoldText("foobar");
  EXPECT_EQ(P.asEscapedMarkdown(), "\\*\\*foo bar\\*\\* \\*\\*foobar\\*\\*");
  EXPECT_EQ(P.asMarkdown(), "**foo bar** **foobar**");
  EXPECT_EQ(P.asPlainText(), "**foo bar** **foobar**");
}

TEST(Paragraph, EmphasizedText) {
  Paragraph P;
  P.appendEmphasizedText("");
  EXPECT_EQ(P.asEscapedMarkdown(), "");
  EXPECT_EQ(P.asMarkdown(), "");
  EXPECT_EQ(P.asPlainText(), "");

  P.appendEmphasizedText(" \n foo\nbar\n ");
  EXPECT_EQ(P.asEscapedMarkdown(), "\\*foo bar\\*");
  EXPECT_EQ(P.asMarkdown(), "*foo bar*");
  EXPECT_EQ(P.asPlainText(), "*foo bar*");

  P.appendSpace().appendEmphasizedText("foobar");
  EXPECT_EQ(P.asEscapedMarkdown(), "\\*foo bar\\* \\*foobar\\*");
  EXPECT_EQ(P.asMarkdown(), "*foo bar* *foobar*");
  EXPECT_EQ(P.asPlainText(), "*foo bar* *foobar*");
}

TEST(Document, Separators) {
  Document D;
  D.addParagraph().appendText("foo");
  D.addCodeBlock("test");
  D.addParagraph().appendText("bar");

  const char ExpectedMarkdown[] = R"md(foo

```cpp
test
```

bar)md";
  EXPECT_EQ(D.asEscapedMarkdown(), ExpectedMarkdown);
  EXPECT_EQ(D.asMarkdown(), ExpectedMarkdown);

  const char ExpectedText[] = R"pt(foo

test

bar)pt";
  EXPECT_EQ(D.asPlainText(), ExpectedText);
}

TEST(Document, Ruler) {
  Document D;
  D.addParagraph().appendText("foo");
  D.addRuler();

  // Ruler followed by paragraph.
  D.addParagraph().appendText("bar");
  EXPECT_EQ(D.asEscapedMarkdown(), "foo\n\n---\nbar");
  EXPECT_EQ(D.asMarkdown(), "foo\n\n---\nbar");
  EXPECT_EQ(D.asPlainText(), "foo\n\nbar");

  D = Document();
  D.addParagraph().appendText("foo");
  D.addRuler();
  D.addCodeBlock("bar");
  // Ruler followed by a codeblock.
  EXPECT_EQ(D.asEscapedMarkdown(), "foo\n\n---\n```cpp\nbar\n```");
  EXPECT_EQ(D.asMarkdown(), "foo\n\n---\n```cpp\nbar\n```");
  EXPECT_EQ(D.asPlainText(), "foo\n\nbar");

  // Ruler followed by another ruler
  D = Document();
  D.addParagraph().appendText("foo");
  D.addRuler();
  D.addRuler();
  EXPECT_EQ(D.asEscapedMarkdown(), "foo");
  EXPECT_EQ(D.asMarkdown(), "foo");
  EXPECT_EQ(D.asPlainText(), "foo");

  // Multiple rulers between blocks
  D.addRuler();
  D.addParagraph().appendText("foo");
  EXPECT_EQ(D.asEscapedMarkdown(), "foo\n\n---\nfoo");
  EXPECT_EQ(D.asMarkdown(), "foo\n\n---\nfoo");
  EXPECT_EQ(D.asPlainText(), "foo\n\nfoo");
}

TEST(Document, Append) {
  Document D;
  D.addParagraph().appendText("foo");
  D.addRuler();
  Document E;
  E.addRuler();
  E.addParagraph().appendText("bar");
  D.append(std::move(E));
  EXPECT_EQ(D.asEscapedMarkdown(), "foo\n\n---\nbar");
  EXPECT_EQ(D.asMarkdown(), "foo\n\n---\nbar");
}

TEST(Document, Heading) {
  Document D;
  D.addHeading(1).appendText("foo");
  D.addHeading(2).appendText("bar");
  D.addParagraph().appendText("baz");
  EXPECT_EQ(D.asEscapedMarkdown(), "# foo\n\n## bar\n\nbaz");
  EXPECT_EQ(D.asMarkdown(), "# foo\n\n## bar\n\nbaz");
  EXPECT_EQ(D.asPlainText(), "foo\n\nbar\n\nbaz");
}

TEST(CodeBlock, Render) {
  Document D;
  // Code blocks preserves any extra spaces.
  D.addCodeBlock("foo\n  bar\n  baz");

  llvm::StringRef ExpectedMarkdown =
      R"md(```cpp
foo
  bar
  baz
```)md";
  llvm::StringRef ExpectedPlainText =
      R"pt(foo
  bar
  baz)pt";
  EXPECT_EQ(D.asEscapedMarkdown(), ExpectedMarkdown);
  EXPECT_EQ(D.asMarkdown(), ExpectedMarkdown);
  EXPECT_EQ(D.asPlainText(), ExpectedPlainText);
  D.addCodeBlock("foo");
  ExpectedMarkdown =
      R"md(```cpp
foo
  bar
  baz
```

```cpp
foo
```)md";
  EXPECT_EQ(D.asEscapedMarkdown(), ExpectedMarkdown);
  EXPECT_EQ(D.asMarkdown(), ExpectedMarkdown);
  ExpectedPlainText =
      R"pt(foo
  bar
  baz

foo)pt";
  EXPECT_EQ(D.asPlainText(), ExpectedPlainText);

  Document D2;
  D2.addCodeBlock("");
  EXPECT_EQ(D2.asEscapedMarkdown(), "```cpp\n```");
  EXPECT_EQ(D2.asMarkdown(), "```cpp\n```");
  EXPECT_EQ(D2.asPlainText(), "");
}

TEST(BulletList, Render) {
  BulletList L;
  // Flat list
  L.addItem().addParagraph().appendText("foo");
  EXPECT_EQ(L.asEscapedMarkdown(), "- foo");
  EXPECT_EQ(L.asMarkdown(), "- foo");
  EXPECT_EQ(L.asPlainText(), "- foo");

  L.addItem().addParagraph().appendText("bar");
  llvm::StringRef Expected = R"md(- foo
- bar)md";
  EXPECT_EQ(L.asEscapedMarkdown(), Expected);
  EXPECT_EQ(L.asMarkdown(), Expected);
  EXPECT_EQ(L.asPlainText(), Expected);

  // Nested list, with a single item.
  Document &D = L.addItem();
  // First item with 2 paragraphs - foo\n\n  baz
  D.addParagraph().appendText("foo");
  D.addParagraph().appendText("baz");

  // Nest one level.
  Document &Inner = D.addBulletList().addItem();
  Inner.addParagraph().appendText("foo");

  // Nest one more level.
  BulletList &InnerList = Inner.addBulletList();
  // Single item, baz\nbaz
  Document &DeepDoc = InnerList.addItem();
  DeepDoc.addParagraph().appendText("baz");
  DeepDoc.addParagraph().appendText("baz");
  StringRef ExpectedMarkdown = R"md(- foo
- bar
- foo

  baz

  - foo

    - baz

      baz)md";
  EXPECT_EQ(L.asEscapedMarkdown(), ExpectedMarkdown);
  EXPECT_EQ(L.asMarkdown(), ExpectedMarkdown);
  StringRef ExpectedPlainText = R"pt(- foo
- bar
- foo

  baz

  - foo

    - baz

      baz)pt";
  EXPECT_EQ(L.asPlainText(), ExpectedPlainText);

  // Termination
  Inner.addParagraph().appendText("after");
  ExpectedMarkdown = R"md(- foo
- bar
- foo

  baz

  - foo

    - baz

      baz

    after)md";
  EXPECT_EQ(L.asEscapedMarkdown(), ExpectedMarkdown);
  EXPECT_EQ(L.asMarkdown(), ExpectedMarkdown);
  ExpectedPlainText = R"pt(- foo
- bar
- foo

  baz

  - foo

    - baz

      baz

    after)pt";
  EXPECT_EQ(L.asPlainText(), ExpectedPlainText);
}

} // namespace
} // namespace markup
} // namespace clangd
} // namespace clang
