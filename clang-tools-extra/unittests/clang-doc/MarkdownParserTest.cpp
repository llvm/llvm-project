//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "markdown/Markdown.h"
#include "gtest/gtest.h"

using namespace clang::doc::markdown;
using namespace llvm;

namespace {

/// Returns the text of the first TextNode child in the given inline list.
static llvm::StringRef firstChildText(const InlineList &L) {
  return llvm::cast<TextNode>(L.front()).getText();
}

TEST(MarkdownNodeTest, TextNode) {
  TextNode N("hello");
  EXPECT_EQ(N.getKind(), NodeKind::NK_Text);
  EXPECT_EQ(N.getText(), "hello");
}

TEST(MarkdownNodeTest, FencedCodeNode) {
  FencedCodeNode N("cpp", "int x = 0;\nint y = 1;");
  EXPECT_EQ(N.getKind(), NodeKind::NK_FencedCode);
  EXPECT_EQ(N.getLang(), "cpp");
  llvm::SmallVector<llvm::StringRef> Lines;
  N.getCode().split(Lines, '\n');
  EXPECT_EQ(Lines[0], "int x = 0;");
  EXPECT_EQ(Lines[1], "int y = 1;");
}

TEST(MarkdownNodeTest, HeadingNode) {
  HeadingNode N(2);
  EXPECT_EQ(N.getKind(), NodeKind::NK_Heading);
  EXPECT_EQ(N.getLevel(), 2u);
}

TEST(MarkdownNodeTest, ThematicBreakNode) {
  ThematicBreakNode N;
  EXPECT_EQ(N.getKind(), NodeKind::NK_ThematicBreak);
}

TEST(MarkdownNodeTest, InlineCodeNode) {
  InlineCodeNode N("foo()");
  EXPECT_EQ(N.getKind(), NodeKind::NK_InlineCode);
  EXPECT_EQ(N.getCode(), "foo()");
}

TEST(MarkdownNodeTest, EmphasisNode) {
  EmphasisNode N;
  TextNode Child("emphasized");
  N.addChild(Child);
  EXPECT_EQ(N.getKind(), NodeKind::NK_Emphasis);
  EXPECT_FALSE(N.children().empty());
  EXPECT_EQ(firstChildText(N.children()), "emphasized");
}

TEST(MarkdownNodeTest, EmphasisRemoveChild) {
  EmphasisNode N;
  TextNode Child("temp");
  N.addChild(Child);
  EXPECT_FALSE(N.children().empty());
  N.removeChild(Child);
  EXPECT_TRUE(N.children().empty());
}

TEST(MarkdownNodeTest, UnorderedListNode) {
  UnorderedListNode N;
  EXPECT_EQ(N.getKind(), NodeKind::NK_UnorderedList);
  EXPECT_TRUE(N.items().empty());
}

TEST(MarkdownNodeTest, UnorderedListRemoveItem) {
  UnorderedListNode List;
  ListItemNode Item;
  List.addItem(Item);
  EXPECT_FALSE(List.items().empty());
  List.removeItem(Item);
  EXPECT_TRUE(List.items().empty());
}

TEST(MarkdownNodeTest, ParagraphNode) {
  ParagraphNode N;
  EXPECT_EQ(N.getKind(), NodeKind::NK_Paragraph);
  EXPECT_TRUE(N.children().empty());
}

TEST(MarkdownNodeTest, DocumentNode) {
  DocumentNode N;
  EXPECT_EQ(N.getKind(), NodeKind::NK_Document);
  EXPECT_TRUE(N.children().empty());
}

TEST(MarkdownNodeTest, ParagraphWithChildren) {
  ParagraphNode Para;
  TextNode Child("hello");
  Para.addChild(Child);
  EXPECT_FALSE(Para.children().empty());
  EXPECT_EQ(firstChildText(Para.children()), "hello");
}

TEST(MarkdownNodeTest, UnorderedListWithItems) {
  UnorderedListNode List;
  ListItemNode Item;
  TextNode Child("item text");
  Item.addChild(Child);
  List.addItem(Item);
  EXPECT_FALSE(List.items().empty());
  EXPECT_EQ(firstChildText(List.items().front().children()), "item text");
}

} // namespace
