//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "support/Markdown.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/Casting.h"
#include "gtest/gtest.h"

using namespace clang::doc::markdown;
using namespace llvm;

namespace {

struct MarkdownParserTest : public ::testing::Test {
  llvm::BumpPtrAllocator Arena;
};

TEST_F(MarkdownParserTest, EmptyInput) {
  auto Nodes = parseMarkdown("", Arena);
  EXPECT_TRUE(Nodes.empty());
}

TEST_F(MarkdownParserTest, WhitespaceOnlyInput) {
  auto Nodes = parseMarkdown("   \n  \n", Arena);
  EXPECT_TRUE(Nodes.empty());
}

TEST_F(MarkdownParserTest, PlainText) {
  auto Nodes = parseMarkdown("hello world", Arena);
  ASSERT_EQ(Nodes.size(), 1u);
  auto *P = cast<ParagraphNode>(Nodes[0]);
  ASSERT_EQ(P->Children.size(), 1u);
  EXPECT_EQ(cast<TextNode>(P->Children[0])->Text, "hello world");
}

TEST_F(MarkdownParserTest, FencedCodeBlock) {
  auto Nodes = parseMarkdown(R"(```cpp
int x = 0;
````````)",
                             Arena);
  ASSERT_EQ(Nodes.size(), 1u);
  auto *N = cast<FencedCodeNode>(Nodes[0]);
  EXPECT_EQ(N->Lang, "cpp");
  ASSERT_EQ(N->Lines.size(), 1u);
}

TEST_F(MarkdownParserTest, FencedCodeBlockNoLang) {
  auto Nodes = parseMarkdown(R"(```
some code
```````)",
                             Arena);
  ASSERT_EQ(Nodes.size(), 1u);
  auto *N = cast<FencedCodeNode>(Nodes[0]);
  EXPECT_TRUE(N->Lang.empty());
}

TEST_F(MarkdownParserTest, UnterminatedFenceProducesCodeNode) {
  auto Nodes = parseMarkdown(R"(```cpp
int x = 0;)",
                             Arena);
  // An unterminated fence should not crash. The parser falls back to emitting a
  // FencedCodeNode with whatever lines were found before the end of input.
  ASSERT_EQ(Nodes.size(), 1u);
  auto *N = cast<FencedCodeNode>(Nodes[0]);
  EXPECT_EQ(N->Lang, "cpp");
  ASSERT_EQ(N->Lines.size(), 1u);
  EXPECT_EQ(N->Lines[0], "int x = 0;");
}

TEST_F(MarkdownParserTest, PipeTable) {
  auto Nodes = parseMarkdown(R"(| A | B |
|---|---|
| 1 | 2 |)",
                             Arena);
  ASSERT_EQ(Nodes.size(), 1u);
  auto *T = cast<TableNode>(Nodes[0]);
  ASSERT_EQ(T->Header.Cells.size(), 2u);
  ASSERT_EQ(T->Header.Cells[0].Children.size(), 1u);
  EXPECT_EQ(cast<TextNode>(T->Header.Cells[0].Children[0])->Text, "A");
  ASSERT_EQ(T->Header.Cells[1].Children.size(), 1u);
  EXPECT_EQ(cast<TextNode>(T->Header.Cells[1].Children[0])->Text, "B");
  ASSERT_EQ(T->Body.size(), 1u);
  ASSERT_EQ(T->Body[0].Cells.size(), 2u);
  EXPECT_EQ(cast<TextNode>(T->Body[0].Cells[0].Children[0])->Text, "1");
  EXPECT_EQ(cast<TextNode>(T->Body[0].Cells[1].Children[0])->Text, "2");
}

// A table cell's text runs through the inline parser, so emphasis inside a cell
// becomes an EmphasisNode rather than literal text.
TEST_F(MarkdownParserTest, TableCellWithEmphasis) {
  auto Nodes = parseMarkdown(R"(| *a* | b |
|---|---|
| c | d |)",
                             Arena);
  ASSERT_EQ(Nodes.size(), 1u);
  auto *T = cast<TableNode>(Nodes[0]);
  ASSERT_EQ(T->Header.Cells.size(), 2u);
  ASSERT_EQ(T->Header.Cells[0].Children.size(), 1u);
  auto *Em = cast<EmphasisNode>(T->Header.Cells[0].Children[0]);
  ASSERT_EQ(Em->Children.size(), 1u);
  EXPECT_EQ(cast<TextNode>(Em->Children[0])->Text, "a");
  EXPECT_EQ(cast<TextNode>(T->Header.Cells[1].Children[0])->Text, "b");
}

// A code span inside a table cell becomes an InlineCodeNode.
TEST_F(MarkdownParserTest, TableCellWithInlineCode) {
  auto Nodes = parseMarkdown(R"(| `x` | y |
|---|---|
| z | w |)",
                             Arena);
  ASSERT_EQ(Nodes.size(), 1u);
  auto *T = cast<TableNode>(Nodes[0]);
  ASSERT_EQ(T->Header.Cells.size(), 2u);
  ASSERT_EQ(T->Header.Cells[0].Children.size(), 1u);
  EXPECT_EQ(cast<InlineCodeNode>(T->Header.Cells[0].Children[0])->Code, "x");
  EXPECT_EQ(cast<TextNode>(T->Body[0].Cells[0].Children[0])->Text, "z");
}

TEST_F(MarkdownParserTest, PipeCharacterWithoutSepRowIsPlainText) {
  auto Nodes = parseMarkdown(R"(a | b
c | d)",
                             Arena);
  // No separator row so should not be parsed as a table.
  for (const auto *Node : Nodes)
    EXPECT_FALSE(isa<TableNode>(Node));
}

TEST_F(MarkdownParserTest, UnorderedList) {
  auto Nodes = parseMarkdown(R"(- foo
- bar
- baz)",
                             Arena);
  ASSERT_EQ(Nodes.size(), 1u);
  auto *N = cast<UnorderedListNode>(Nodes[0]);
  ASSERT_EQ(N->Items.size(), 3u);
  // Each item's children are the inline nodes from parseInline.
  StringRef ExpectedText[] = {"foo", "bar", "baz"};
  for (size_t I = 0; I < N->Items.size(); ++I) {
    auto *Item = N->Items[I];
    ASSERT_EQ(Item->Children.size(), 1u);
    EXPECT_EQ(cast<TextNode>(Item->Children[0])->Text, ExpectedText[I]);
  }
}

TEST_F(MarkdownParserTest, ListItemWithEmphasis) {
  auto Nodes = parseMarkdown("- an *important* note", Arena);
  ASSERT_EQ(Nodes.size(), 1u);
  auto *N = cast<UnorderedListNode>(Nodes[0]);
  ASSERT_EQ(N->Items.size(), 1u);
  auto *Item = N->Items[0];
  ASSERT_EQ(Item->Children.size(), 3u);
  EXPECT_EQ(cast<TextNode>(Item->Children[0])->Text, "an ");
  auto *Em = cast<EmphasisNode>(Item->Children[1]);
  ASSERT_EQ(Em->Children.size(), 1u);
  EXPECT_EQ(cast<TextNode>(Em->Children[0])->Text, "important");
  EXPECT_EQ(cast<TextNode>(Item->Children[2])->Text, " note");
}

TEST_F(MarkdownParserTest, OrderedList) {
  auto Nodes = parseMarkdown(R"(1. foo
2. bar
3. baz)",
                             Arena);
  ASSERT_EQ(Nodes.size(), 1u);
  auto *N = cast<OrderedListNode>(Nodes[0]);
  EXPECT_EQ(N->Start, 1u);
  ASSERT_EQ(N->Items.size(), 3u);
  StringRef ExpectedText[] = {"foo", "bar", "baz"};
  for (size_t I = 0; I < N->Items.size(); ++I) {
    auto *Item = N->Items[I];
    ASSERT_EQ(Item->Children.size(), 1u);
    EXPECT_EQ(cast<TextNode>(Item->Children[0])->Text, ExpectedText[I]);
  }
}

TEST_F(MarkdownParserTest, OrderedListCustomStart) {
  auto Nodes = parseMarkdown(R"(5. five
6. six)",
                             Arena);
  ASSERT_EQ(Nodes.size(), 1u);
  auto *N = cast<OrderedListNode>(Nodes[0]);
  EXPECT_EQ(N->Start, 5u);
  ASSERT_EQ(N->Items.size(), 2u);
  EXPECT_EQ(cast<TextNode>(N->Items[0]->Children[0])->Text, "five");
  EXPECT_EQ(cast<TextNode>(N->Items[1]->Children[0])->Text, "six");
}

TEST_F(MarkdownParserTest, OrderedListItemWithEmphasis) {
  auto Nodes = parseMarkdown("1. an *important* note", Arena);
  ASSERT_EQ(Nodes.size(), 1u);
  auto *N = cast<OrderedListNode>(Nodes[0]);
  EXPECT_EQ(N->Start, 1u);
  ASSERT_EQ(N->Items.size(), 1u);
  auto *Item = N->Items[0];
  ASSERT_EQ(Item->Children.size(), 3u);
  EXPECT_EQ(cast<TextNode>(Item->Children[0])->Text, "an ");
  auto *Em = cast<EmphasisNode>(Item->Children[1]);
  ASSERT_EQ(Em->Children.size(), 1u);
  EXPECT_EQ(cast<TextNode>(Em->Children[0])->Text, "important");
  EXPECT_EQ(cast<TextNode>(Item->Children[2])->Text, " note");
}

TEST_F(MarkdownParserTest, MixedContent) {
  auto Nodes = parseMarkdown(R"(some text
```````
code
````````
- item)",
                             Arena);
  ASSERT_EQ(Nodes.size(), 3u);
  EXPECT_TRUE(isa<ParagraphNode>(Nodes[0]));
  EXPECT_TRUE(isa<FencedCodeNode>(Nodes[1]));
  EXPECT_TRUE(isa<UnorderedListNode>(Nodes[2]));
}

// CommonMark §4.5 example 120: tilde fences work the same as backtick fences.
TEST_F(MarkdownParserTest, TildeFence) {
  auto Nodes = parseMarkdown(R"(~~~
int x = 0;
~~~)",
                             Arena);
  ASSERT_EQ(Nodes.size(), 1u);
  auto *N = cast<FencedCodeNode>(Nodes[0]);
  EXPECT_TRUE(N->Lang.empty());
  ASSERT_EQ(N->Lines.size(), 1u);
}

// CommonMark §4.5 example 120: tilde fence with a language tag.
TEST_F(MarkdownParserTest, TildeFenceWithLang) {
  auto Nodes = parseMarkdown(R"(~~~cpp
int x = 0;
~~~)",
                             Arena);
  ASSERT_EQ(Nodes.size(), 1u);
  auto *N = cast<FencedCodeNode>(Nodes[0]);
  EXPECT_EQ(N->Lang, "cpp");
  ASSERT_EQ(N->Lines.size(), 1u);
}

// CommonMark §4.5 example 122: a tilde line does not close a backtick fence.
TEST_F(MarkdownParserTest, ClosingFenceMustMatchOpeningChar) {
  auto Nodes = parseMarkdown(R"(```
aaa
~~~
````````)",
                             Arena);
  ASSERT_EQ(Nodes.size(), 1u);
  auto *N = cast<FencedCodeNode>(Nodes[0]);
  // ~~~ is content, not a closing fence.
  ASSERT_EQ(N->Lines.size(), 2u);
}

// CommonMark §4.5 example 130: a code block can be empty.
TEST_F(MarkdownParserTest, EmptyFencedCodeBlock) {
  auto Nodes = parseMarkdown(R"(```
```````)",
                             Arena);
  ASSERT_EQ(Nodes.size(), 1u);
  auto *N = cast<FencedCodeNode>(Nodes[0]);
  EXPECT_TRUE(N->Lines.empty());
}

// CommonMark §4.5 example 129: a code block may contain only blank lines.
TEST_F(MarkdownParserTest, FencedCodeBlockBlankLineContent) {
  auto Nodes = parseMarkdown("```\n\n  \n```", Arena);
  ASSERT_EQ(Nodes.size(), 1u);
  auto *N = cast<FencedCodeNode>(Nodes[0]);
  ASSERT_EQ(N->Lines.size(), 2u);
}

// CommonMark §4.5 example 142: lang tag is captured from the info string.
TEST_F(MarkdownParserTest, InfoStringLangTag) {
  auto Nodes = parseMarkdown(R"(```ruby
def foo(x)
  return 3
end
``````)",
                             Arena);
  ASSERT_EQ(Nodes.size(), 1u);
  auto *N = cast<FencedCodeNode>(Nodes[0]);
  EXPECT_EQ(N->Lang, "ruby");
  ASSERT_EQ(N->Lines.size(), 3u);
}

// CommonMark §4.5 example 146: tilde fence info string may contain backticks.
TEST_F(MarkdownParserTest, TildeFenceInfoStringWithBackticks) {
  auto Nodes = parseMarkdown(R"(~~~ aa ``` ~~~
foo
~~~)",
                             Arena);
  ASSERT_EQ(Nodes.size(), 1u);
  auto *N = cast<FencedCodeNode>(Nodes[0]);
  EXPECT_EQ(N->Lang, "aa ``` ~~~");
  ASSERT_EQ(N->Lines.size(), 1u);
}

// CommonMark §4.5 example 124: the closing fence must be at least as long as
// the opening fence. Our parser closes on the first line with 3 matching fence
// chars regardless of opening length, so this documents the current
// non-conformant behavior.
// TODO: fix as part of the CommonMark TODO in parseMarkdown().
TEST_F(MarkdownParserTest, ClosingFenceLengthTODO) {
  auto Nodes = parseMarkdown("````\naaa\n```", Arena);
  ASSERT_EQ(Nodes.size(), 1u);
  auto *N = cast<FencedCodeNode>(Nodes[0]);
  ASSERT_EQ(N->Lines.size(), 1u);
}

TEST_F(MarkdownParserTest, EmphasisAsterisk) {
  auto Nodes = parseMarkdown("an *important* word", Arena);
  ASSERT_EQ(Nodes.size(), 1u);
  auto *P = cast<ParagraphNode>(Nodes[0]);
  ASSERT_EQ(P->Children.size(), 3u);
  EXPECT_EQ(cast<TextNode>(P->Children[0])->Text, "an ");
  auto *Em = cast<EmphasisNode>(P->Children[1]);
  ASSERT_EQ(Em->Children.size(), 1u);
  EXPECT_EQ(cast<TextNode>(Em->Children[0])->Text, "important");
  EXPECT_EQ(cast<TextNode>(P->Children[2])->Text, " word");
}

TEST_F(MarkdownParserTest, EmphasisUnderscore) {
  auto Nodes = parseMarkdown("_em_", Arena);
  ASSERT_EQ(Nodes.size(), 1u);
  auto *P = cast<ParagraphNode>(Nodes[0]);
  ASSERT_EQ(P->Children.size(), 1u);
  auto *Em = cast<EmphasisNode>(P->Children[0]);
  ASSERT_EQ(Em->Children.size(), 1u);
  EXPECT_EQ(cast<TextNode>(Em->Children[0])->Text, "em");
}

TEST_F(MarkdownParserTest, StrongAsterisk) {
  auto Nodes = parseMarkdown("**bold**", Arena);
  ASSERT_EQ(Nodes.size(), 1u);
  auto *P = cast<ParagraphNode>(Nodes[0]);
  ASSERT_EQ(P->Children.size(), 1u);
  auto *St = cast<StrongNode>(P->Children[0]);
  ASSERT_EQ(St->Children.size(), 1u);
  EXPECT_EQ(cast<TextNode>(St->Children[0])->Text, "bold");
}

TEST_F(MarkdownParserTest, StrongUnderscore) {
  auto Nodes = parseMarkdown("__bold__", Arena);
  ASSERT_EQ(Nodes.size(), 1u);
  auto *P = cast<ParagraphNode>(Nodes[0]);
  ASSERT_EQ(P->Children.size(), 1u);
  auto *St = cast<StrongNode>(P->Children[0]);
  ASSERT_EQ(St->Children.size(), 1u);
  EXPECT_EQ(cast<TextNode>(St->Children[0])->Text, "bold");
}

// Two delimiters must be parsed as strong, not as nested emphasis.
TEST_F(MarkdownParserTest, StrongBindsBeforeEmphasis) {
  auto Nodes = parseMarkdown("**strong**", Arena);
  ASSERT_EQ(Nodes.size(), 1u);
  auto *P = cast<ParagraphNode>(Nodes[0]);
  ASSERT_EQ(P->Children.size(), 1u);
  EXPECT_TRUE(isa<StrongNode>(P->Children[0]));
}

TEST_F(MarkdownParserTest, InlineCode) {
  auto Nodes = parseMarkdown("call `foo()` here", Arena);
  ASSERT_EQ(Nodes.size(), 1u);
  auto *P = cast<ParagraphNode>(Nodes[0]);
  ASSERT_EQ(P->Children.size(), 3u);
  EXPECT_EQ(cast<TextNode>(P->Children[0])->Text, "call ");
  EXPECT_EQ(cast<InlineCodeNode>(P->Children[1])->Code, "foo()");
  EXPECT_EQ(cast<TextNode>(P->Children[2])->Text, " here");
}

// CommonMark §6.1: a doubled backtick fence lets the span contain a single
// backtick.
TEST_F(MarkdownParserTest, InlineCodeDoubleBacktick) {
  auto Nodes = parseMarkdown("``a`b``", Arena);
  ASSERT_EQ(Nodes.size(), 1u);
  auto *P = cast<ParagraphNode>(Nodes[0]);
  ASSERT_EQ(P->Children.size(), 1u);
  EXPECT_EQ(cast<InlineCodeNode>(P->Children[0])->Code, "a`b");
}

// Emphasis and strong recurse, so a code span inside emphasis is parsed.
TEST_F(MarkdownParserTest, CodeSpanInsideEmphasis) {
  auto Nodes = parseMarkdown("*see `x`*", Arena);
  ASSERT_EQ(Nodes.size(), 1u);
  auto *P = cast<ParagraphNode>(Nodes[0]);
  ASSERT_EQ(P->Children.size(), 1u);
  auto *Em = cast<EmphasisNode>(P->Children[0]);
  ASSERT_EQ(Em->Children.size(), 2u);
  EXPECT_EQ(cast<TextNode>(Em->Children[0])->Text, "see ");
  EXPECT_EQ(cast<InlineCodeNode>(Em->Children[1])->Code, "x");
}

TEST_F(MarkdownParserTest, CodeSpanInsideStrong) {
  auto Nodes = parseMarkdown("**a `b`**", Arena);
  ASSERT_EQ(Nodes.size(), 1u);
  auto *P = cast<ParagraphNode>(Nodes[0]);
  ASSERT_EQ(P->Children.size(), 1u);
  auto *St = cast<StrongNode>(P->Children[0]);
  ASSERT_EQ(St->Children.size(), 2u);
  EXPECT_EQ(cast<TextNode>(St->Children[0])->Text, "a ");
  EXPECT_EQ(cast<InlineCodeNode>(St->Children[1])->Code, "b");
}

// A delimiter with whitespace on the inside does not open emphasis.
TEST_F(MarkdownParserTest, UnmatchedDelimiterIsText) {
  auto Nodes = parseMarkdown("a * b", Arena);
  ASSERT_EQ(Nodes.size(), 1u);
  auto *P = cast<ParagraphNode>(Nodes[0]);
  ASSERT_EQ(P->Children.size(), 1u);
  EXPECT_EQ(cast<TextNode>(P->Children[0])->Text, "a * b");
}

// An unterminated code span leaves the backtick as literal text.
TEST_F(MarkdownParserTest, UnterminatedCodeSpanIsText) {
  auto Nodes = parseMarkdown("a `b c", Arena);
  ASSERT_EQ(Nodes.size(), 1u);
  auto *P = cast<ParagraphNode>(Nodes[0]);
  ASSERT_EQ(P->Children.size(), 1u);
  EXPECT_EQ(cast<TextNode>(P->Children[0])->Text, "a `b c");
}

// Inline parsing must not disturb plain text with no markers.
TEST_F(MarkdownParserTest, PlainTextHasNoInlineNodes) {
  auto Nodes = parseMarkdown("just words", Arena);
  ASSERT_EQ(Nodes.size(), 1u);
  auto *P = cast<ParagraphNode>(Nodes[0]);
  ASSERT_EQ(P->Children.size(), 1u);
  EXPECT_EQ(cast<TextNode>(P->Children[0])->Text, "just words");
}

TEST_F(MarkdownParserTest, Heading1) {
  auto Nodes = parseMarkdown("# Title", Arena);
  ASSERT_EQ(Nodes.size(), 1u);
  auto *H = cast<HeadingNode>(Nodes[0]);
  EXPECT_EQ(H->Level, 1u);
  ASSERT_EQ(H->Children.size(), 1u);
  EXPECT_EQ(cast<TextNode>(H->Children[0])->Text, "Title");
}

TEST_F(MarkdownParserTest, Heading2) {
  auto Nodes = parseMarkdown("## Title", Arena);
  ASSERT_EQ(Nodes.size(), 1u);
  auto *H = cast<HeadingNode>(Nodes[0]);
  EXPECT_EQ(H->Level, 2u);
  ASSERT_EQ(H->Children.size(), 1u);
  EXPECT_EQ(cast<TextNode>(H->Children[0])->Text, "Title");
}

TEST_F(MarkdownParserTest, Heading3) {
  auto Nodes = parseMarkdown("### Title", Arena);
  ASSERT_EQ(Nodes.size(), 1u);
  auto *H = cast<HeadingNode>(Nodes[0]);
  EXPECT_EQ(H->Level, 3u);
  ASSERT_EQ(H->Children.size(), 1u);
  EXPECT_EQ(cast<TextNode>(H->Children[0])->Text, "Title");
}

TEST_F(MarkdownParserTest, HeadingWithInlineCode) {
  auto Nodes = parseMarkdown("# Use `foo()`", Arena);
  ASSERT_EQ(Nodes.size(), 1u);
  auto *H = cast<HeadingNode>(Nodes[0]);
  EXPECT_EQ(H->Level, 1u);
  ASSERT_EQ(H->Children.size(), 2u);
  EXPECT_EQ(cast<TextNode>(H->Children[0])->Text, "Use ");
  EXPECT_EQ(cast<InlineCodeNode>(H->Children[1])->Code, "foo()");
}

TEST_F(MarkdownParserTest, HeadingWithEmphasis) {
  auto Nodes = parseMarkdown("## see *this*", Arena);
  ASSERT_EQ(Nodes.size(), 1u);
  auto *H = cast<HeadingNode>(Nodes[0]);
  EXPECT_EQ(H->Level, 2u);
  ASSERT_EQ(H->Children.size(), 2u);
  EXPECT_EQ(cast<TextNode>(H->Children[0])->Text, "see ");
  auto *Em = cast<EmphasisNode>(H->Children[1]);
  ASSERT_EQ(Em->Children.size(), 1u);
  EXPECT_EQ(cast<TextNode>(Em->Children[0])->Text, "this");
}

// Seven or more # characters are not a valid ATX heading, so the line falls
// back to a plain-text paragraph.
TEST_F(MarkdownParserTest, SevenHashesIsPlainText) {
  auto Nodes = parseMarkdown("####### too many", Arena);
  ASSERT_EQ(Nodes.size(), 1u);
  auto *P = cast<ParagraphNode>(Nodes[0]);
  ASSERT_EQ(P->Children.size(), 1u);
  EXPECT_EQ(cast<TextNode>(P->Children[0])->Text, "####### too many");
}

TEST_F(MarkdownParserTest, ThematicBreakDashes) {
  auto Nodes = parseMarkdown("---", Arena);
  ASSERT_EQ(Nodes.size(), 1u);
  EXPECT_TRUE(isa<ThematicBreakNode>(Nodes[0]));
}

TEST_F(MarkdownParserTest, ThematicBreakAsterisks) {
  auto Nodes = parseMarkdown("***", Arena);
  ASSERT_EQ(Nodes.size(), 1u);
  EXPECT_TRUE(isa<ThematicBreakNode>(Nodes[0]));
}

TEST_F(MarkdownParserTest, ThematicBreakUnderscores) {
  auto Nodes = parseMarkdown("___", Arena);
  ASSERT_EQ(Nodes.size(), 1u);
  EXPECT_TRUE(isa<ThematicBreakNode>(Nodes[0]));
}

//===----------------------------------------------------------------------===//
// CommonMark spec edge cases (spec.commonmark.org/0.31.2). Each test cites the
// section and example it exercises. Cases marked DIVERGENCE document where this
// simplified parser intentionally differs from full CommonMark.
//===----------------------------------------------------------------------===//

// CommonMark §4.1 Example 51: spaces are allowed between the characters.
TEST_F(MarkdownParserTest, ThematicBreakSpacedDashes) {
  auto Nodes = parseMarkdown("- - -", Arena);
  ASSERT_EQ(Nodes.size(), 1u);
  EXPECT_TRUE(isa<ThematicBreakNode>(Nodes[0]));
}

// CommonMark §4.1 Example 44: +++ is not a thematic break.
TEST_F(MarkdownParserTest, PlusesAreNotThematicBreak) {
  auto Nodes = parseMarkdown("+++", Arena);
  ASSERT_EQ(Nodes.size(), 1u);
  auto *P = cast<ParagraphNode>(Nodes[0]);
  ASSERT_EQ(P->Children.size(), 1u);
  EXPECT_EQ(cast<TextNode>(P->Children[0])->Text, "+++");
}

// CommonMark §4.1 Example 46: fewer than three characters is not a break.
TEST_F(MarkdownParserTest, TwoDashesAreNotThematicBreak) {
  auto Nodes = parseMarkdown("--", Arena);
  ASSERT_EQ(Nodes.size(), 1u);
  auto *P = cast<ParagraphNode>(Nodes[0]);
  ASSERT_EQ(P->Children.size(), 1u);
  EXPECT_EQ(cast<TextNode>(P->Children[0])->Text, "--");
}

// CommonMark §4.2 Example 64: a # not followed by a space is not a heading.
TEST_F(MarkdownParserTest, HashWithoutSpaceIsNotHeading) {
  auto Nodes = parseMarkdown("#5 bolt", Arena);
  ASSERT_EQ(Nodes.size(), 1u);
  auto *P = cast<ParagraphNode>(Nodes[0]);
  ASSERT_EQ(P->Children.size(), 1u);
  EXPECT_EQ(cast<TextNode>(P->Children[0])->Text, "#5 bolt");
}

// CommonMark §4.2 Example 64: "#hashtag" is a paragraph, not a heading.
TEST_F(MarkdownParserTest, HashtagIsNotHeading) {
  auto Nodes = parseMarkdown("#hashtag", Arena);
  ASSERT_EQ(Nodes.size(), 1u);
  auto *P = cast<ParagraphNode>(Nodes[0]);
  ASSERT_EQ(P->Children.size(), 1u);
  EXPECT_EQ(cast<TextNode>(P->Children[0])->Text, "#hashtag");
}

// CommonMark §4.2 Example 67: spaces around the heading content are stripped.
TEST_F(MarkdownParserTest, HeadingStripsContentSpaces) {
  auto Nodes = parseMarkdown("#         foo", Arena);
  ASSERT_EQ(Nodes.size(), 1u);
  auto *H = cast<HeadingNode>(Nodes[0]);
  EXPECT_EQ(H->Level, 1u);
  ASSERT_EQ(H->Children.size(), 1u);
  EXPECT_EQ(cast<TextNode>(H->Children[0])->Text, "foo");
}

// CommonMark §5.2: * is a valid bullet list marker.
TEST_F(MarkdownParserTest, UnorderedListAsteriskMarker) {
  auto Nodes = parseMarkdown("* foo", Arena);
  ASSERT_EQ(Nodes.size(), 1u);
  auto *N = cast<UnorderedListNode>(Nodes[0]);
  ASSERT_EQ(N->Items.size(), 1u);
  EXPECT_EQ(cast<TextNode>(N->Items[0]->Children[0])->Text, "foo");
}

// CommonMark §5.2 Example 301: + is a valid bullet list marker.
TEST_F(MarkdownParserTest, UnorderedListPlusMarker) {
  auto Nodes = parseMarkdown("+ foo", Arena);
  ASSERT_EQ(Nodes.size(), 1u);
  auto *N = cast<UnorderedListNode>(Nodes[0]);
  ASSERT_EQ(N->Items.size(), 1u);
  EXPECT_EQ(cast<TextNode>(N->Items[0]->Children[0])->Text, "foo");
}

// CommonMark §5.2 Example 267: an ordered list may start at 0.
TEST_F(MarkdownParserTest, OrderedListStartZero) {
  auto Nodes = parseMarkdown("0. ok", Arena);
  ASSERT_EQ(Nodes.size(), 1u);
  auto *N = cast<OrderedListNode>(Nodes[0]);
  EXPECT_EQ(N->Start, 0u);
  ASSERT_EQ(N->Items.size(), 1u);
  EXPECT_EQ(cast<TextNode>(N->Items[0]->Children[0])->Text, "ok");
}

// CommonMark §5.2 Example 296: ordered lists may use a ) delimiter. DIVERGENCE:
// this parser only recognizes the . delimiter, so "1) foo" is plain text.
TEST_F(MarkdownParserTest, OrderedListParenDelimiterNotSupported) {
  auto Nodes = parseMarkdown("1) foo", Arena);
  ASSERT_EQ(Nodes.size(), 1u);
  auto *P = cast<ParagraphNode>(Nodes[0]);
  ASSERT_EQ(P->Children.size(), 1u);
  EXPECT_EQ(cast<TextNode>(P->Children[0])->Text, "1) foo");
}

// CommonMark §6.2 Example 355: intraword emphasis with asterisks.
TEST_F(MarkdownParserTest, IntrawordEmphasisAsterisk) {
  auto Nodes = parseMarkdown("foo*bar*", Arena);
  ASSERT_EQ(Nodes.size(), 1u);
  auto *P = cast<ParagraphNode>(Nodes[0]);
  ASSERT_EQ(P->Children.size(), 2u);
  EXPECT_EQ(cast<TextNode>(P->Children[0])->Text, "foo");
  auto *Em = cast<EmphasisNode>(P->Children[1]);
  ASSERT_EQ(Em->Children.size(), 1u);
  EXPECT_EQ(cast<TextNode>(Em->Children[0])->Text, "bar");
}

// CommonMark §6.2 Example 381: intraword strong with asterisks.
TEST_F(MarkdownParserTest, IntrawordStrongAsterisk) {
  auto Nodes = parseMarkdown("foo**bar**", Arena);
  ASSERT_EQ(Nodes.size(), 1u);
  auto *P = cast<ParagraphNode>(Nodes[0]);
  ASSERT_EQ(P->Children.size(), 2u);
  EXPECT_EQ(cast<TextNode>(P->Children[0])->Text, "foo");
  auto *St = cast<StrongNode>(P->Children[1]);
  ASSERT_EQ(St->Children.size(), 1u);
  EXPECT_EQ(cast<TextNode>(St->Children[0])->Text, "bar");
}

// CommonMark §6.2 Example 360: intraword underscores do not open or close
// emphasis, so "foo_bar_" stays as literal text.
TEST_F(MarkdownParserTest, IntrawordUnderscoreIsText) {
  auto Nodes = parseMarkdown("foo_bar_", Arena);
  ASSERT_EQ(Nodes.size(), 1u);
  auto *P = cast<ParagraphNode>(Nodes[0]);
  ASSERT_EQ(P->Children.size(), 1u);
  EXPECT_EQ(cast<TextNode>(P->Children[0])->Text, "foo_bar_");
}

// CommonMark §6.1 Example 331: a code span strips one leading and trailing
// space when both are present.
TEST_F(MarkdownParserTest, CodeSpanStripsSurroundingSpaces) {
  auto Nodes = parseMarkdown("`` x ``", Arena);
  ASSERT_EQ(Nodes.size(), 1u);
  auto *P = cast<ParagraphNode>(Nodes[0]);
  ASSERT_EQ(P->Children.size(), 1u);
  EXPECT_EQ(cast<InlineCodeNode>(P->Children[0])->Code, "x");
}

// CommonMark §6.2 Example 413: a triple run splits across two matches, the
// inner pair forming strong and the outer pair emphasis, so "***foo***" is
// emphasis wrapping strong. The old findClosingDelim search matched a run only
// against an equal-length run and could not split one this way.
TEST_F(MarkdownParserTest, TripleDelimiterBoldItalic) {
  auto Nodes = parseMarkdown("***foo***", Arena);
  ASSERT_EQ(Nodes.size(), 1u);
  auto *P = cast<ParagraphNode>(Nodes[0]);
  ASSERT_EQ(P->Children.size(), 1u);
  auto *Em = cast<EmphasisNode>(P->Children[0]);
  ASSERT_EQ(Em->Children.size(), 1u);
  auto *St = cast<StrongNode>(Em->Children[0]);
  ASSERT_EQ(St->Children.size(), 1u);
  EXPECT_EQ(cast<TextNode>(St->Children[0])->Text, "foo");
}

// CommonMark §6.2: emphasis containing a strong span, "*foo **bar** baz*". The
// outer emphasis spans delimiter runs of two different lengths, which the
// equal-length findClosingDelim search could not pair.
TEST_F(MarkdownParserTest, MixedDelimitersEmStrongEm) {
  auto Nodes = parseMarkdown("*foo **bar** baz*", Arena);
  ASSERT_EQ(Nodes.size(), 1u);
  auto *P = cast<ParagraphNode>(Nodes[0]);
  ASSERT_EQ(P->Children.size(), 1u);
  auto *Em = cast<EmphasisNode>(P->Children[0]);
  ASSERT_EQ(Em->Children.size(), 3u);
  EXPECT_EQ(cast<TextNode>(Em->Children[0])->Text, "foo ");
  auto *St = cast<StrongNode>(Em->Children[1]);
  ASSERT_EQ(St->Children.size(), 1u);
  EXPECT_EQ(cast<TextNode>(St->Children[0])->Text, "bar");
  EXPECT_EQ(cast<TextNode>(Em->Children[2])->Text, " baz");
}

// CommonMark §6.2: strong containing emphasis with text on both sides,
// "**foo *bar* baz**". The inner emphasis closes before the outer strong does,
// which the single forward scan handled only by coincidence of nesting order.
TEST_F(MarkdownParserTest, NestedEmphasisInsideStrong) {
  auto Nodes = parseMarkdown("**foo *bar* baz**", Arena);
  ASSERT_EQ(Nodes.size(), 1u);
  auto *P = cast<ParagraphNode>(Nodes[0]);
  ASSERT_EQ(P->Children.size(), 1u);
  auto *St = cast<StrongNode>(P->Children[0]);
  ASSERT_EQ(St->Children.size(), 3u);
  EXPECT_EQ(cast<TextNode>(St->Children[0])->Text, "foo ");
  auto *Em = cast<EmphasisNode>(St->Children[1]);
  ASSERT_EQ(Em->Children.size(), 1u);
  EXPECT_EQ(cast<TextNode>(Em->Children[0])->Text, "bar");
  EXPECT_EQ(cast<TextNode>(St->Children[2])->Text, " baz");
}

// CommonMark §6.2 rule of three: when a closer can also open, it may not match
// an opener whose run length sums with the closer's to a multiple of three. In
// "**foo*bar*" the leading ** cannot close against the inner *, so ** stays
// literal and only *bar* becomes emphasis.
TEST_F(MarkdownParserTest, MultipleOfThreeBlocksClose) {
  auto Nodes = parseMarkdown("**foo*bar*", Arena);
  ASSERT_EQ(Nodes.size(), 1u);
  auto *P = cast<ParagraphNode>(Nodes[0]);
  ASSERT_EQ(P->Children.size(), 2u);
  EXPECT_EQ(cast<TextNode>(P->Children[0])->Text, "**foo");
  auto *Em = cast<EmphasisNode>(P->Children[1]);
  ASSERT_EQ(Em->Children.size(), 1u);
  EXPECT_EQ(cast<TextNode>(Em->Children[0])->Text, "bar");
}

TEST_F(MarkdownParserTest, BlockQuote) {
  auto Nodes = parseMarkdown("> hello", Arena);
  ASSERT_EQ(Nodes.size(), 1u);
  auto *Q = cast<BlockQuoteNode>(Nodes[0]);
  ASSERT_EQ(Q->Children.size(), 1u);
  auto *P = cast<ParagraphNode>(Q->Children[0]);
  ASSERT_EQ(P->Children.size(), 1u);
  EXPECT_EQ(cast<TextNode>(P->Children[0])->Text, "hello");
}

TEST_F(MarkdownParserTest, BlockQuoteWithFencedCode) {
  auto Nodes = parseMarkdown(R"(> ```cpp
> int x = 0;
> ```)",
                             Arena);
  ASSERT_EQ(Nodes.size(), 1u);
  auto *Q = cast<BlockQuoteNode>(Nodes[0]);
  ASSERT_EQ(Q->Children.size(), 1u);
  auto *Code = cast<FencedCodeNode>(Q->Children[0]);
  EXPECT_EQ(Code->Lang, "cpp");
  ASSERT_EQ(Code->Lines.size(), 1u);
  EXPECT_EQ(Code->Lines[0], "int x = 0;");
}

TEST_F(MarkdownParserTest, BlockQuoteWithEmphasis) {
  auto Nodes = parseMarkdown("> an *important* note", Arena);
  ASSERT_EQ(Nodes.size(), 1u);
  auto *Q = cast<BlockQuoteNode>(Nodes[0]);
  ASSERT_EQ(Q->Children.size(), 1u);
  auto *P = cast<ParagraphNode>(Q->Children[0]);
  ASSERT_EQ(P->Children.size(), 3u);
  EXPECT_EQ(cast<TextNode>(P->Children[0])->Text, "an ");
  auto *Em = cast<EmphasisNode>(P->Children[1]);
  ASSERT_EQ(Em->Children.size(), 1u);
  EXPECT_EQ(cast<TextNode>(Em->Children[0])->Text, "important");
  EXPECT_EQ(cast<TextNode>(P->Children[2])->Text, " note");
}

TEST_F(MarkdownParserTest, NestedBlockQuote) {
  auto Nodes = parseMarkdown("> > deep", Arena);
  ASSERT_EQ(Nodes.size(), 1u);
  auto *Outer = cast<BlockQuoteNode>(Nodes[0]);
  ASSERT_EQ(Outer->Children.size(), 1u);
  auto *Inner = cast<BlockQuoteNode>(Outer->Children[0]);
  ASSERT_EQ(Inner->Children.size(), 1u);
  auto *P = cast<ParagraphNode>(Inner->Children[0]);
  ASSERT_EQ(P->Children.size(), 1u);
  EXPECT_EQ(cast<TextNode>(P->Children[0])->Text, "deep");
}

} // namespace
