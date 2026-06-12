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
  auto *N = cast<TextNode>(Nodes[0]);
  EXPECT_EQ(N->Text, "hello world");
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

TEST_F(MarkdownParserTest, UnterminatedFenceReturnsEmpty) {
  auto Nodes = parseMarkdown(R"(```cpp
int x = 0;)",
                             Arena);
  // Unterminated fence should not crash and should produce a code node
  // with whatever lines were found.
  EXPECT_FALSE(Nodes.empty());
}

TEST_F(MarkdownParserTest, PipeTable) {
  auto Nodes = parseMarkdown(R"(| A | B |
|---|---|
| 1 | 2 |)",
                             Arena);
  ASSERT_EQ(Nodes.size(), 1u);
  EXPECT_TRUE(isa<TableNode>(Nodes[0]));
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
  EXPECT_EQ(cast<TextNode>(N->Items[0]->Children[0])->Text, "foo");
  EXPECT_EQ(cast<TextNode>(N->Items[1]->Children[0])->Text, "bar");
  EXPECT_EQ(cast<TextNode>(N->Items[2]->Children[0])->Text, "baz");
}

TEST_F(MarkdownParserTest, MixedContent) {
  auto Nodes = parseMarkdown(R"(some text
```````
code
````````
- item)",
                             Arena);
  EXPECT_EQ(Nodes.size(), 3u);
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

// CommonMark §4.5 example 124: closing fence must be at least as long as the
// opening fence.
// TODO: our parser currently closes on the first line with 3 matching fence
// chars regardless of opening fence length. Fix as part of the CommonMark
// TODO in parseMarkdown().
TEST_F(MarkdownParserTest, ClosingFenceLengthTODO) {
  auto Nodes = parseMarkdown("````\naaa\n```", Arena);
  // The ``` line should not close the ```` fence per CommonMark, but our
  // parser currently treats it as a closing fence. This test documents the
  // current (non-conformant) behavior.
  ASSERT_EQ(Nodes.size(), 1u);
  auto *N = cast<FencedCodeNode>(Nodes[0]);
  ASSERT_EQ(N->Lines.size(), 1u);
}

TEST_F(MarkdownParserTest, EmphasisAsterisk) {
  auto Nodes = parseMarkdown("an *important* word", Arena);
  ASSERT_EQ(Nodes.size(), 3u);
  EXPECT_EQ(cast<TextNode>(Nodes[0])->Text, "an ");
  auto *Em = cast<EmphasisNode>(Nodes[1]);
  ASSERT_EQ(Em->Children.size(), 1u);
  EXPECT_EQ(cast<TextNode>(Em->Children[0])->Text, "important");
  EXPECT_EQ(cast<TextNode>(Nodes[2])->Text, " word");
}

TEST_F(MarkdownParserTest, EmphasisUnderscore) {
  auto Nodes = parseMarkdown("_em_", Arena);
  ASSERT_EQ(Nodes.size(), 1u);
  auto *Em = cast<EmphasisNode>(Nodes[0]);
  ASSERT_EQ(Em->Children.size(), 1u);
  EXPECT_EQ(cast<TextNode>(Em->Children[0])->Text, "em");
}

TEST_F(MarkdownParserTest, StrongAsterisk) {
  auto Nodes = parseMarkdown("**bold**", Arena);
  ASSERT_EQ(Nodes.size(), 1u);
  auto *St = cast<StrongNode>(Nodes[0]);
  ASSERT_EQ(St->Children.size(), 1u);
  EXPECT_EQ(cast<TextNode>(St->Children[0])->Text, "bold");
}

TEST_F(MarkdownParserTest, StrongUnderscore) {
  auto Nodes = parseMarkdown("__bold__", Arena);
  ASSERT_EQ(Nodes.size(), 1u);
  auto *St = cast<StrongNode>(Nodes[0]);
  ASSERT_EQ(St->Children.size(), 1u);
  EXPECT_EQ(cast<TextNode>(St->Children[0])->Text, "bold");
}

// Two delimiters must be parsed as strong, not as nested emphasis.
TEST_F(MarkdownParserTest, StrongBindsBeforeEmphasis) {
  auto Nodes = parseMarkdown("**strong**", Arena);
  ASSERT_EQ(Nodes.size(), 1u);
  EXPECT_TRUE(isa<StrongNode>(Nodes[0]));
}

TEST_F(MarkdownParserTest, InlineCode) {
  auto Nodes = parseMarkdown("call `foo()` here", Arena);
  ASSERT_EQ(Nodes.size(), 3u);
  EXPECT_EQ(cast<TextNode>(Nodes[0])->Text, "call ");
  EXPECT_EQ(cast<InlineCodeNode>(Nodes[1])->Code, "foo()");
  EXPECT_EQ(cast<TextNode>(Nodes[2])->Text, " here");
}

// CommonMark §6.1: a doubled backtick fence lets the span contain a single
// backtick.
TEST_F(MarkdownParserTest, InlineCodeDoubleBacktick) {
  auto Nodes = parseMarkdown("``a`b``", Arena);
  ASSERT_EQ(Nodes.size(), 1u);
  EXPECT_EQ(cast<InlineCodeNode>(Nodes[0])->Code, "a`b");
}

// Emphasis and strong recurse, so a code span inside emphasis is parsed.
TEST_F(MarkdownParserTest, CodeSpanInsideEmphasis) {
  auto Nodes = parseMarkdown("*see `x`*", Arena);
  ASSERT_EQ(Nodes.size(), 1u);
  auto *Em = cast<EmphasisNode>(Nodes[0]);
  ASSERT_EQ(Em->Children.size(), 2u);
  EXPECT_EQ(cast<TextNode>(Em->Children[0])->Text, "see ");
  EXPECT_EQ(cast<InlineCodeNode>(Em->Children[1])->Code, "x");
}

TEST_F(MarkdownParserTest, CodeSpanInsideStrong) {
  auto Nodes = parseMarkdown("**a `b`**", Arena);
  ASSERT_EQ(Nodes.size(), 1u);
  auto *St = cast<StrongNode>(Nodes[0]);
  ASSERT_EQ(St->Children.size(), 2u);
  EXPECT_EQ(cast<TextNode>(St->Children[0])->Text, "a ");
  EXPECT_EQ(cast<InlineCodeNode>(St->Children[1])->Code, "b");
}

// A delimiter with whitespace on the inside does not open emphasis.
TEST_F(MarkdownParserTest, UnmatchedDelimiterIsText) {
  auto Nodes = parseMarkdown("a * b", Arena);
  ASSERT_EQ(Nodes.size(), 1u);
  EXPECT_EQ(cast<TextNode>(Nodes[0])->Text, "a * b");
}

// An unterminated code span leaves the backtick as literal text.
TEST_F(MarkdownParserTest, UnterminatedCodeSpanIsText) {
  auto Nodes = parseMarkdown("a `b c", Arena);
  ASSERT_EQ(Nodes.size(), 1u);
  EXPECT_EQ(cast<TextNode>(Nodes[0])->Text, "a `b c");
}

// Inline parsing must not disturb plain text with no markers.
TEST_F(MarkdownParserTest, PlainTextHasNoInlineNodes) {
  auto Nodes = parseMarkdown("just words", Arena);
  ASSERT_EQ(Nodes.size(), 1u);
  EXPECT_EQ(cast<TextNode>(Nodes[0])->Text, "just words");
}

} // namespace