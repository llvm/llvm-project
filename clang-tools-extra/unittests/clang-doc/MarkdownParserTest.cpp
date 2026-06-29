//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "support/Markdown.h"
#include "gtest/gtest.h"

using namespace clang::doc::markdown;
using namespace llvm;

namespace {

TEST(MarkdownNodeTest, TextNode) {
  TextNode N("hello");
  EXPECT_EQ(N.Kind, NodeKind::NK_Text);
  EXPECT_EQ(N.getText(), "hello");
}

TEST(MarkdownNodeTest, FencedCodeNode) {
  FencedCodeNode N("cpp", "int x = 0;\nint y = 1;");
  EXPECT_EQ(N.Kind, NodeKind::NK_FencedCode);
  EXPECT_EQ(N.getLang(), "cpp");
  EXPECT_TRUE(N.getCode().contains("int x = 0;"));
  EXPECT_TRUE(N.getCode().contains("int y = 1;"));
}

TEST(MarkdownNodeTest, HeadingNode) {
  HeadingNode N(2);
  EXPECT_EQ(N.Kind, NodeKind::NK_Heading);
  EXPECT_EQ(N.getLevel(), 2u);
}

TEST(MarkdownNodeTest, ThematicBreakNode) {
  ThematicBreakNode N;
  EXPECT_EQ(N.Kind, NodeKind::NK_ThematicBreak);
}

TEST(MarkdownNodeTest, InlineCodeNode) {
  InlineCodeNode N("foo()");
  EXPECT_EQ(N.Kind, NodeKind::NK_InlineCode);
  EXPECT_EQ(N.getCode(), "foo()");
}

TEST(MarkdownNodeTest, EmphasisNode) {
  EmphasisNode N;
  TextNode Child("emphasized");
  N.Children.push_back(Child);
  EXPECT_EQ(N.Kind, NodeKind::NK_Emphasis);
  EXPECT_FALSE(N.Children.empty());
  EXPECT_EQ(llvm::cast<TextNode>(N.Children.front()).getText(), "emphasized");
}

TEST(MarkdownNodeTest, UnorderedListNode) {
  UnorderedListNode N;
  EXPECT_EQ(N.Kind, NodeKind::NK_UnorderedList);
  EXPECT_TRUE(N.Items.empty());
}

TEST(MarkdownNodeTest, ParagraphNode) {
  ParagraphNode N;
  EXPECT_EQ(N.Kind, NodeKind::NK_Paragraph);
  EXPECT_TRUE(N.Children.empty());
}

TEST(MarkdownNodeTest, DocumentNode) {
  DocumentNode N;
  EXPECT_EQ(N.Kind, NodeKind::NK_Document);
  EXPECT_TRUE(N.Children.empty());
}

TEST(MarkdownNodeTest, ParagraphWithChildren) {
  ParagraphNode Para;
  TextNode Child("hello");
  Para.Children.push_back(Child);
  EXPECT_FALSE(Para.Children.empty());
  EXPECT_EQ(llvm::cast<TextNode>(Para.Children.front()).getText(), "hello");
}

TEST(MarkdownNodeTest, UnorderedListWithItems) {
  UnorderedListNode List;
  ListItemNode Item;
  TextNode Child("item text");
  Item.Children.push_back(Child);
  List.Items.push_back(Item);
  EXPECT_FALSE(List.Items.empty());
  EXPECT_EQ(llvm::cast<TextNode>(List.Items.front().Children.front()).getText(),
            "item text");
}

} // namespace
