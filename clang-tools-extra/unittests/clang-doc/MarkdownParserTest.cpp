#include "support/Markdown.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/Casting.h"
#include "gtest/gtest.h"

using namespace clang::doc::markdown;
using namespace llvm;

namespace {

TEST(MarkdownNodeTest, TextNode) {
  BumpPtrAllocator Arena;
  auto *N = new (Arena) TextNode("hello");
  EXPECT_EQ(N->Kind, NodeKind::NK_Text);
  EXPECT_EQ(N->Text, "hello");
  EXPECT_TRUE(isa<TextNode>(N));
}

TEST(MarkdownNodeTest, FencedCodeNode) {
  BumpPtrAllocator Arena;
  StringRef Lines[] = {"int x = 0;"};
  auto *N = new (Arena) FencedCodeNode("cpp", ArrayRef(Lines));
  EXPECT_EQ(N->Kind, NodeKind::NK_FencedCode);
  EXPECT_EQ(N->Lang, "cpp");
  EXPECT_EQ(N->Lines.size(), 1u);
  EXPECT_TRUE(isa<FencedCodeNode>(N));
}

TEST(MarkdownNodeTest, HeadingNode) {
  BumpPtrAllocator Arena;
  auto *N = new (Arena) HeadingNode(2, {});
  EXPECT_EQ(N->Kind, NodeKind::NK_Heading);
  EXPECT_EQ(N->Level, 2u);
  EXPECT_TRUE(isa<HeadingNode>(N));
}

TEST(MarkdownNodeTest, ThematicBreakNode) {
  BumpPtrAllocator Arena;
  auto *N = new (Arena) ThematicBreakNode();
  EXPECT_EQ(N->Kind, NodeKind::NK_ThematicBreak);
  EXPECT_TRUE(isa<ThematicBreakNode>(N));
}

TEST(MarkdownNodeTest, InlineCodeNode) {
  BumpPtrAllocator Arena;
  auto *N = new (Arena) InlineCodeNode("foo()");
  EXPECT_EQ(N->Kind, NodeKind::NK_InlineCode);
  EXPECT_EQ(N->Code, "foo()");
  EXPECT_TRUE(isa<InlineCodeNode>(N));
}

TEST(MarkdownNodeTest, EmphasisNode) {
  BumpPtrAllocator Arena;
  auto *N = new (Arena) EmphasisNode({});
  EXPECT_EQ(N->Kind, NodeKind::NK_Emphasis);
  EXPECT_TRUE(isa<EmphasisNode>(N));
}

TEST(MarkdownNodeTest, UnorderedListNode) {
  BumpPtrAllocator Arena;
  auto *N = new (Arena) UnorderedListNode({});
  EXPECT_EQ(N->Kind, NodeKind::NK_UnorderedList);
  EXPECT_TRUE(isa<UnorderedListNode>(N));
}

TEST(MarkdownNodeTest, ParagraphNode) {
  BumpPtrAllocator Arena;
  auto *N = new (Arena) ParagraphNode({});
  EXPECT_EQ(N->Kind, NodeKind::NK_Paragraph);
  EXPECT_TRUE(isa<ParagraphNode>(N));
}

} // namespace