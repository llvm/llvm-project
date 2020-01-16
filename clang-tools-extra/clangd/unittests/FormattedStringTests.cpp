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
#include "llvm/Support/raw_ostream.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace clang {
namespace clangd {
namespace markup {
namespace {

TEST(Render, Escaping) {
  // Check some ASCII punctuation
  Paragraph P;
  P.appendText("*!`");
  EXPECT_EQ(P.asMarkdown(), "\\*\\!\\`");

  // Check all ASCII punctuation.
  P = Paragraph();
  std::string Punctuation = R"txt(!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~)txt";
  // Same text, with each character escaped.
  std::string EscapedPunctuation;
  EscapedPunctuation.reserve(2 * Punctuation.size());
  for (char C : Punctuation)
    EscapedPunctuation += std::string("\\") + C;
  P.appendText(Punctuation);
  EXPECT_EQ(P.asMarkdown(), EscapedPunctuation);

  // In code blocks we don't need to escape ASCII punctuation.
  P = Paragraph();
  P.appendCode("* foo !+ bar * baz");
  EXPECT_EQ(P.asMarkdown(), "`* foo !+ bar * baz`");

  // But we have to escape the backticks.
  P = Paragraph();
  P.appendCode("foo`bar`baz");
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

  P.appendText("after");
  EXPECT_EQ(P.asMarkdown(), "after");
  EXPECT_EQ(P.asPlainText(), "after");

  P.appendCode("foobar");
  EXPECT_EQ(P.asMarkdown(), "after `foobar`");
  EXPECT_EQ(P.asPlainText(), "after foobar");

  P.appendText("bat");
  EXPECT_EQ(P.asMarkdown(), "after `foobar` bat");
  EXPECT_EQ(P.asPlainText(), "after foobar bat");
}

TEST(Paragraph, ExtraSpaces) {
  // Make sure spaces inside chunks are dropped.
  Paragraph P;
  P.appendText("foo\n   \t   baz");
  P.appendCode(" bar\n");
  EXPECT_EQ(P.asMarkdown(), "foo baz `bar`");
  EXPECT_EQ(P.asPlainText(), "foo baz bar");
}

TEST(Paragraph, NewLines) {
  // New lines before and after chunks are dropped.
  Paragraph P;
  P.appendText(" \n foo\nbar\n ");
  P.appendCode(" \n foo\nbar \n ");
  EXPECT_EQ(P.asMarkdown(), "foo bar `foo bar`");
  EXPECT_EQ(P.asPlainText(), "foo bar foo bar");
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
  EXPECT_EQ(D.asMarkdown(), "foo  \n\n---\nbar");
  EXPECT_EQ(D.asPlainText(), "foo\n\nbar");

  D = Document();
  D.addParagraph().appendText("foo");
  D.addRuler();
  D.addCodeBlock("bar");
  // Ruler followed by a codeblock.
  EXPECT_EQ(D.asMarkdown(), "foo  \n\n---\n```cpp\nbar\n```");
  EXPECT_EQ(D.asPlainText(), "foo\n\nbar");

  // Ruler followed by another ruler
  D = Document();
  D.addParagraph().appendText("foo");
  D.addRuler();
  D.addRuler();
  EXPECT_EQ(D.asMarkdown(), "foo");
  EXPECT_EQ(D.asPlainText(), "foo");

  // Multiple rulers between blocks
  D.addRuler();
  D.addParagraph().appendText("foo");
  EXPECT_EQ(D.asMarkdown(), "foo  \n\n---\nfoo");
  EXPECT_EQ(D.asPlainText(), "foo\n\nfoo");
}

TEST(Document, Heading) {
  Document D;
  D.addHeading(1).appendText("foo");
  D.addHeading(2).appendText("bar");
  D.addParagraph().appendText("baz");
  EXPECT_EQ(D.asMarkdown(), "# foo  \n## bar  \nbaz");
  EXPECT_EQ(D.asPlainText(), "foo\nbar\nbaz");
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
  EXPECT_EQ(D.asMarkdown(), ExpectedMarkdown);
  ExpectedPlainText =
      R"pt(foo
  bar
  baz

foo)pt";
  EXPECT_EQ(D.asPlainText(), ExpectedPlainText);
}

TEST(BulletList, Render) {
  BulletList L;
  // Flat list
  L.addItem().addParagraph().appendText("foo");
  EXPECT_EQ(L.asMarkdown(), "- foo");
  EXPECT_EQ(L.asPlainText(), "- foo");

  L.addItem().addParagraph().appendText("bar");
  llvm::StringRef Expected = R"md(- foo
- bar)md";
  EXPECT_EQ(L.asMarkdown(), Expected);
  EXPECT_EQ(L.asPlainText(), Expected);

  // Nested list, with a single item.
  Document &D = L.addItem();
  // First item with foo\nbaz
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
