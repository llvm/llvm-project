//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "support/Markdown.h"
#include "llvm/Support/Allocator.h"
#include "gtest/gtest.h"

using namespace clang::doc::markdown;

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
  const auto &N = Nodes[0];
  EXPECT_EQ(N.Kind, NodeKind::NK_Text);
  EXPECT_EQ(N.Content, "hello world");
}

TEST_F(MarkdownParserTest, FencedCodeBlock) {
  auto Nodes = parseMarkdown(R"(```cpp
int x = 0;
````)",
                             Arena);
  ASSERT_EQ(Nodes.size(), 1u);
  const auto &N = Nodes[0];
  EXPECT_EQ(N.Kind, NodeKind::NK_FencedCode);
  EXPECT_EQ(N.Content, "cpp");
  ASSERT_EQ(N.Children.size(), 1u);
}

TEST_F(MarkdownParserTest, FencedCodeBlockNoLang) {
  auto Nodes = parseMarkdown(R"(```
some code
```)",
                             Arena);
  ASSERT_EQ(Nodes.size(), 1u);
  const auto &N = Nodes[0];
  EXPECT_EQ(N.Kind, NodeKind::NK_FencedCode);
  EXPECT_TRUE(N.Content.empty());
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
  EXPECT_EQ(Nodes[0].Kind, NodeKind::NK_Table);
}

TEST_F(MarkdownParserTest, PipeCharacterWithoutSepRowIsPlainText) {
  auto Nodes = parseMarkdown(R"(a | b
c | d)",
                             Arena);
  // No separator row so should not be parsed as a table.
  for (const auto &Node : Nodes)
    EXPECT_NE(Node.Kind, NodeKind::NK_Table);
}

TEST_F(MarkdownParserTest, UnorderedList) {
  auto Nodes = parseMarkdown(R"(- foo
- bar
- baz)",
                             Arena);
  ASSERT_EQ(Nodes.size(), 1u);
  const auto &N = Nodes[0];
  EXPECT_EQ(N.Kind, NodeKind::NK_UnorderedList);
  ASSERT_EQ(N.Children.size(), 3u);
  EXPECT_EQ(N.Children[0].Content, "foo");
  EXPECT_EQ(N.Children[1].Content, "bar");
  EXPECT_EQ(N.Children[2].Content, "baz");
}

TEST_F(MarkdownParserTest, MixedContent) {
  auto Nodes = parseMarkdown(R"(some text
```
code
````
- item)",
                             Arena);
  EXPECT_EQ(Nodes.size(), 3u);
}

} // namespace