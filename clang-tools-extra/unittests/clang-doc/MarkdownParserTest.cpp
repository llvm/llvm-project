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

TEST(MarkdownParserTest, EmptyInput) {
  llvm::BumpPtrAllocator Arena;
  auto Nodes = parseMarkdown("", Arena);
  EXPECT_TRUE(Nodes.empty());
}

TEST(MarkdownParserTest, WhitespaceOnlyInput) {
  llvm::BumpPtrAllocator Arena;
  auto Nodes = parseMarkdown("   \n  \n", Arena);
  EXPECT_TRUE(Nodes.empty());
}

TEST(MarkdownParserTest, PlainText) {
  llvm::BumpPtrAllocator Arena;
  auto Nodes = parseMarkdown("hello world", Arena);
  ASSERT_EQ(Nodes.size(), 1u);
  EXPECT_EQ(Nodes[0].Kind, NodeKind::NK_Text);
  EXPECT_EQ(Nodes[0].Content, "hello world");
}

TEST(MarkdownParserTest, FencedCodeBlock) {
  llvm::BumpPtrAllocator Arena;
  auto Nodes = parseMarkdown("```cpp\nint x = 0;\n```", Arena);
  ASSERT_EQ(Nodes.size(), 1u);
  EXPECT_EQ(Nodes[0].Kind, NodeKind::NK_FencedCode);
  EXPECT_EQ(Nodes[0].Content, "cpp");
  ASSERT_EQ(Nodes[0].Children.size(), 1u);
}

TEST(MarkdownParserTest, FencedCodeBlockNoLang) {
  llvm::BumpPtrAllocator Arena;
  auto Nodes = parseMarkdown("```\nsome code\n```", Arena);
  ASSERT_EQ(Nodes.size(), 1u);
  EXPECT_EQ(Nodes[0].Kind, NodeKind::NK_FencedCode);
  EXPECT_TRUE(Nodes[0].Content.empty());
}

TEST(MarkdownParserTest, UnterminatedFenceReturnsEmpty) {
  llvm::BumpPtrAllocator Arena;
  auto Nodes = parseMarkdown("```cpp\nint x = 0;", Arena);
  // Unterminated fence should not crash and should produce a code node
  // with whatever lines were found.
  EXPECT_FALSE(Nodes.empty());
}

TEST(MarkdownParserTest, PipeTable) {
  llvm::BumpPtrAllocator Arena;
  auto Nodes = parseMarkdown("| A | B |\n|---|---|\n| 1 | 2 |", Arena);
  ASSERT_EQ(Nodes.size(), 1u);
  EXPECT_EQ(Nodes[0].Kind, NodeKind::NK_Table);
}

TEST(MarkdownParserTest, PipeCharacterWithoutSepRowIsPlainText) {
  llvm::BumpPtrAllocator Arena;
  auto Nodes = parseMarkdown("a | b\nc | d", Arena);
  // No separator row so should not be parsed as a table
  for (const auto &Node : Nodes)
    EXPECT_NE(Node.Kind, NodeKind::NK_Table);
}

TEST(MarkdownParserTest, UnorderedList) {
  llvm::BumpPtrAllocator Arena;
  auto Nodes = parseMarkdown("- foo\n- bar\n- baz", Arena);
  ASSERT_EQ(Nodes.size(), 1u);
  EXPECT_EQ(Nodes[0].Kind, NodeKind::NK_UnorderedList);
  ASSERT_EQ(Nodes[0].Children.size(), 3u);
  EXPECT_EQ(Nodes[0].Children[0].Content, "foo");
  EXPECT_EQ(Nodes[0].Children[1].Content, "bar");
  EXPECT_EQ(Nodes[0].Children[2].Content, "baz");
}

TEST(MarkdownParserTest, MixedContent) {
  llvm::BumpPtrAllocator Arena;
  auto Nodes = parseMarkdown("some text\n```\ncode\n```\n- item", Arena);
  EXPECT_EQ(Nodes.size(), 3u);
}

} // namespace