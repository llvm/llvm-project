#include "support/Markdown.h"
#include "llvm/Support/Casting.h"
#include "gtest/gtest.h"

using namespace clang::doc::markdown;
using namespace llvm;

namespace {

TEST(MarkdownNodeTest, TextNode) {
  TextNode N("hello");
  EXPECT_EQ(N.Kind, NodeKind::NK_Text);
  EXPECT_EQ(N.Text, "hello");
}

TEST(MarkdownNodeTest, FencedCodeNode) {
  StringRef Lines[] = {"int x = 0;"};
  FencedCodeNode N("cpp", ArrayRef(Lines));
  EXPECT_EQ(N.Kind, NodeKind::NK_FencedCode);
  EXPECT_EQ(N.Lang, "cpp");
  EXPECT_EQ(N.Lines.size(), 1u);
}

TEST(MarkdownNodeTest, HeadingNode) {
  HeadingNode N(2, {});
  EXPECT_EQ(N.Kind, NodeKind::NK_Heading);
  EXPECT_EQ(N.Level, 2u);
}

TEST(MarkdownNodeTest, ThematicBreakNode) {
  ThematicBreakNode N;
  EXPECT_EQ(N.Kind, NodeKind::NK_ThematicBreak);
}

TEST(MarkdownNodeTest, InlineCodeNode) {
  InlineCodeNode N("foo()");
  EXPECT_EQ(N.Kind, NodeKind::NK_InlineCode);
  EXPECT_EQ(N.Code, "foo()");
}

TEST(MarkdownNodeTest, EmphasisNode) {
  EmphasisNode N({});
  EXPECT_EQ(N.Kind, NodeKind::NK_Emphasis);
}

TEST(MarkdownNodeTest, UnorderedListNode) {
  UnorderedListNode N({});
  EXPECT_EQ(N.Kind, NodeKind::NK_UnorderedList);
}

TEST(MarkdownNodeTest, ParagraphNode) {
  ParagraphNode N({});
  EXPECT_EQ(N.Kind, NodeKind::NK_Paragraph);
}

} // namespace