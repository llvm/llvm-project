#include "MDParser.h"
#include "llvm/ADT/StringRef.h"
#include "gtest/gtest.h"

using llvm::StringRef;

namespace clang {
namespace doc {
using namespace clang::doc::md;

TEST(MDParserTest, Strong) {
  ASTContext AST;
  MarkdownParser Parser(AST);
  std::vector<StringRef> Line = {{"**Strong**"}};
  Parser.parse(Line);
  ContainerBlock *Root = AST.getRoot();

  ASSERT_NE(Root, nullptr);
  ASSERT_EQ(Root->Children.front().Type, MDType::Paragraph);
  auto &Paragraph = static_cast<InlineContainerBlock &>(Root->Children.front());
  ASSERT_EQ(Paragraph.Children.front().Type, md::InlineType::Strong);
  auto &Strong = static_cast<InlineContainer &>(Paragraph.Children.front());
  ASSERT_EQ(Strong.Children.front().Type, InlineType::Text);
  auto &Text = static_cast<TextInline &>(Strong.Children.front());
  ASSERT_EQ(Text.Text, "Strong");
}

TEST(MDParserTest, Emphasis) {
  ASTContext AST;
  MarkdownParser Parser(AST);
  std::vector<StringRef> Line = {{"*Emphasis*"}};
  Parser.parse(Line);
  ContainerBlock *Root = AST.getRoot();

  ASSERT_NE(Root, nullptr);
  ASSERT_EQ(Root->Children.front().Type, MDType::Paragraph);
  auto &Paragraph = static_cast<InlineContainerBlock &>(Root->Children.front());
  ASSERT_EQ(Paragraph.Children.front().Type, InlineType::Emphasis);
  auto &Emphasis = static_cast<InlineContainer &>(Paragraph.Children.front());
  ASSERT_EQ(Emphasis.Children.front().Type, InlineType::Text);
  auto &Text = static_cast<TextInline &>(Emphasis.Children.front());
  ASSERT_EQ(Text.Text, "Emphasis");
}

TEST(MDParserTest, Text) {
  ASTContext AST;
  MarkdownParser Parser(AST);
  std::vector<StringRef> Line = {{"Text"}};
  Parser.parse(Line);
  ContainerBlock *Root = AST.getRoot();

  ASSERT_NE(Root, nullptr);
  ASSERT_EQ(Root->Children.front().Type, MDType::Paragraph);
  auto &Paragraph = static_cast<InlineContainerBlock &>(Root->Children.front());
  ASSERT_EQ(Paragraph.Children.front().Type, InlineType::Text);
  auto &Text = static_cast<TextInline &>(Paragraph.Children.front());
  ASSERT_EQ(Text.Text, "Text");
}

// TEST(MDParserTest, StrongEmphasis) {
//   ASTContext AST;
//   MarkdownParser Parser(AST);
//   std::vector<StringRef> Line = {{"***StrongEmphasis***"}};
//   Parser.parse(Line);
//   ContainerBlock *Root = AST.getRoot();
//   ASSERT_NE(Root, nullptr);
//   ASSERT_EQ(Root->Children.front().Type, MDType::Paragraph);
//   auto &Paragraph = static_cast<InlineContainerBlock
//   &>(Root->Children.front()); auto &Emphasis = static_cast<Inline
//   &>(Paragraph.Children.front()); ASSERT_EQ(Emphasis.Type,
//   InlineType::Emphasis); auto &Strong = static_cast<InlineContainer
//   &>(Emphasis).Children.front(); ASSERT_EQ(Strong.Type, InlineType::Strong);
//   auto &Text = static_cast<TextInline &>(
//       static_cast<InlineContainer &>(Strong).Children.front());
//   ASSERT_EQ(Text.Text, "StrongEmphasis");
// }

TEST(MDParserTest, LiteralTextBeforeEmphasis) {
  ASTContext AST;
  MarkdownParser Parser(AST);
  std::vector<StringRef> Line = {{"Literal*Emphasis*"}};
  Parser.parse(Line);
  ContainerBlock *Root = AST.getRoot();

  ASSERT_NE(Root, nullptr);
  ASSERT_EQ(Root->Children.front().Type, MDType::Paragraph);
  auto &Paragraph = static_cast<InlineContainerBlock &>(Root->Children.front());
  auto &Text = static_cast<TextInline &>(Paragraph.Children.front());
  ASSERT_EQ(Text.Type, InlineType::Text);
  ASSERT_EQ(Text.Text, "Literal");
  auto &Emphasis = static_cast<InlineContainer &>(Paragraph.Children.back());
  ASSERT_EQ(Emphasis.Type, InlineType::Emphasis);
  auto EmphasisText = static_cast<TextInline &>(Emphasis.Children.front());
  ASSERT_EQ(EmphasisText.Text, "Emphasis");
}

TEST(MDParserTest, LiteralTextBeforeEmphasisWithSpace) {
  ASTContext AST;
  MarkdownParser Parser(AST);
  std::vector<StringRef> Line = {{"Literal *Emphasis*"}};
  Parser.parse(Line);
  ContainerBlock *Root = AST.getRoot();

  ASSERT_NE(Root, nullptr);
  ASSERT_EQ(Root->Children.front().Type, MDType::Paragraph);
  auto &Paragraph = static_cast<InlineContainerBlock &>(Root->Children.front());
  auto &Text = static_cast<TextInline &>(Paragraph.Children.front());
  ASSERT_EQ(Text.Type, InlineType::Text);
  ASSERT_EQ(Text.Text, "Literal ");
  auto &Emphasis = static_cast<InlineContainer &>(Paragraph.Children.back());
  ASSERT_EQ(Emphasis.Type, InlineType::Emphasis);
  auto EmphasisText = static_cast<TextInline &>(Emphasis.Children.front());
  ASSERT_EQ(EmphasisText.Text, "Emphasis");
}

TEST(MDParserTest, LiteralTextAfterEmphasis) {
  ASTContext AST;
  MarkdownParser Parser(AST);
  std::vector<StringRef> Line = {{"*Emphasis*Literal"}};
  Parser.parse(Line);
  ContainerBlock *Root = AST.getRoot();

  ASSERT_NE(Root, nullptr);
  ASSERT_EQ(Root->Children.front().Type, MDType::Paragraph);
  auto &Paragraph = static_cast<InlineContainerBlock &>(Root->Children.front());
  auto &Emphasis = static_cast<InlineContainer &>(Paragraph.Children.front());
  ASSERT_EQ(Emphasis.Type, InlineType::Emphasis);
  auto EmphasisText = static_cast<TextInline &>(Emphasis.Children.front());
  ASSERT_EQ(EmphasisText.Text, "Emphasis");
  auto &Text = static_cast<TextInline &>(Paragraph.Children.back());
  ASSERT_EQ(Text.Type, InlineType::Text);
  ASSERT_EQ(Text.Text, "Literal");
}

TEST(MDParserTest, LiteralTextAfterEmphasisWithSpace) {
  ASTContext AST;
  MarkdownParser Parser(AST);
  std::vector<StringRef> Line = {{"*Emphasis* Literal"}};
  Parser.parse(Line);
  ContainerBlock *Root = AST.getRoot();

  ASSERT_NE(Root, nullptr);
  ASSERT_EQ(Root->Children.front().Type, MDType::Paragraph);
  auto &Paragraph = static_cast<InlineContainerBlock &>(Root->Children.front());
  auto &Emphasis = static_cast<InlineContainer &>(Paragraph.Children.front());
  ASSERT_EQ(Emphasis.Type, InlineType::Emphasis);
  auto EmphasisText = static_cast<TextInline &>(Emphasis.Children.front());
  ASSERT_EQ(EmphasisText.Text, "Emphasis");
  auto &Text = static_cast<TextInline &>(Paragraph.Children.back());
  ASSERT_EQ(Text.Type, InlineType::Text);
  ASSERT_EQ(Text.Text, " Literal");
}

TEST(MDParserTest, LiteralTextBeforeAndAfterEmphasis) {
  ASTContext AST;
  MarkdownParser Parser(AST);
  std::vector<StringRef> Line = {{"Literal*Emphasis*Literal"}};
  Parser.parse(Line);
  ContainerBlock *Root = AST.getRoot();

  ASSERT_NE(Root, nullptr);
  ASSERT_EQ(Root->Children.front().Type, MDType::Paragraph);
  auto &Paragraph = static_cast<InlineContainerBlock &>(Root->Children.front());
  auto &FrontText = static_cast<TextInline &>(Paragraph.Children.front());
  ASSERT_EQ(FrontText.Type, InlineType::Text);
  ASSERT_EQ(FrontText.Text, "Literal");

  auto &Emphasis = static_cast<InlineContainer &>(
      *(++(Paragraph.Children.front().getIterator())));
  ASSERT_EQ(Emphasis.Type, InlineType::Emphasis);
  auto EmphasisText = static_cast<TextInline &>(Emphasis.Children.front());
  ASSERT_EQ(EmphasisText.Text, "Emphasis");

  auto &BackText = static_cast<TextInline &>(Paragraph.Children.back());
  ASSERT_EQ(BackText.Type, InlineType::Text);
  ASSERT_EQ(BackText.Text, "Literal");
}

// TEST(MDParserTest, LeftoverBeforeStrong) {
//   ASTContext AST;
//   MarkdownParser Parser(AST);
//   std::vector<StringRef> Line = {{"***Leftover**"}};
//   Parser.parse(Line);
//   ContainerBlock *Root = AST.getRoot();
//
//   ASSERT_NE(Root, nullptr);
//   ASSERT_EQ(Root->Children.front().Type, MDType::Paragraph);
//   auto &Paragraph = static_cast<InlineContainerBlock
//   &>(Root->Children.front());
//
//   auto &Leftover = static_cast<TextInline &>(Paragraph.Children.front());
//   ASSERT_EQ(Leftover.Type, InlineType::Text);
//   ASSERT_EQ(Leftover.Text, "*");
//
//   auto &Strong = static_cast<InlineContainer &>(
//       *(++(Paragraph.Children.front().getIterator())));
//   ASSERT_EQ(Strong.Type, InlineType::Strong);
//   auto &StrongText = static_cast<TextInline &>(Strong.Children.front());
//   ASSERT_EQ(StrongText.Text, "Leftover");
// }
//
// TEST(MDParserTest, LeftoverAfterStrong) {
//   ASTContext AST;
//   MarkdownParser Parser(AST);
//   std::vector<StringRef> Line = {{"**Leftover***"}};
//   Parser.parse(Line);
//   ContainerBlock *Root = AST.getRoot();
//
//   ASSERT_NE(Root, nullptr);
//   ASSERT_EQ(Root->Children.front().Type, MDType::Paragraph);
//   auto &Paragraph = static_cast<InlineContainerBlock
//   &>(Root->Children.front());
//
//   auto &Strong = static_cast<InlineContainer &>(Paragraph.Children.front());
//   ASSERT_EQ(Strong.Type, InlineType::Strong);
//   auto &StrongText = static_cast<TextInline &>(Strong.Children.front());
//   ASSERT_EQ(StrongText.Text, "Leftover");
//
//   auto &Leftover = static_cast<TextInline &>(Paragraph.Children.back());
//   ASSERT_EQ(Leftover.Type, InlineType::Text);
//   ASSERT_EQ(Leftover.Text, "*");
// }
} // namespace doc
} // namespace clang
