//===-- CodeCompletionStringsTests.cpp --------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "CodeCompletionStrings.h"
#include "clang/Sema/CodeCompleteConsumer.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace clang {
namespace clangd {
namespace {

class CompletionStringTest : public ::testing::Test {
public:
  CompletionStringTest()
      : Allocator(std::make_shared<clang::GlobalCodeCompletionAllocator>()),
        CCTUInfo(Allocator), Builder(*Allocator, CCTUInfo) {}

protected:
  void labelAndInsertText(const CodeCompletionString &CCS,
                          bool EnableSnippets = false) {
    Label.clear();
    InsertText.clear();
    getLabelAndInsertText(CCS, &Label, &InsertText, EnableSnippets);
  }

  std::shared_ptr<clang::GlobalCodeCompletionAllocator> Allocator;
  CodeCompletionTUInfo CCTUInfo;
  CodeCompletionBuilder Builder;
  std::string Label;
  std::string InsertText;
};

TEST_F(CompletionStringTest, Detail) {
  Builder.AddResultTypeChunk("result");
  Builder.AddResultTypeChunk("redundant result no no");
  EXPECT_EQ(getDetail(*Builder.TakeString()), "result");
}

TEST_F(CompletionStringTest, Documentation) {
  Builder.addBriefComment("This is ignored");
  EXPECT_EQ(formatDocumentation(*Builder.TakeString(), "Is this brief?"),
            "Is this brief?");
}

TEST_F(CompletionStringTest, DocumentationWithAnnotation) {
  Builder.addBriefComment("This is ignored");
  Builder.AddAnnotation("Ano");
  EXPECT_EQ(formatDocumentation(*Builder.TakeString(), "Is this brief?"),
            "Annotation: Ano\n\nIs this brief?");
}

TEST_F(CompletionStringTest, MultipleAnnotations) {
  Builder.AddAnnotation("Ano1");
  Builder.AddAnnotation("Ano2");
  Builder.AddAnnotation("Ano3");

  EXPECT_EQ(formatDocumentation(*Builder.TakeString(), ""),
            "Annotations: Ano1 Ano2 Ano3\n");
}

TEST_F(CompletionStringTest, SimpleLabelAndInsert) {
  Builder.AddTypedTextChunk("X");
  Builder.AddResultTypeChunk("result no no");
  labelAndInsertText(*Builder.TakeString());
  EXPECT_EQ(Label, "X");
  EXPECT_EQ(InsertText, "X");
}

TEST_F(CompletionStringTest, FunctionPlainText) {
  Builder.AddResultTypeChunk("result no no");
  Builder.AddTypedTextChunk("Foo");
  Builder.AddChunk(CodeCompletionString::CK_LeftParen);
  Builder.AddPlaceholderChunk("p1");
  Builder.AddChunk(CodeCompletionString::CK_Comma);
  Builder.AddPlaceholderChunk("p2");
  Builder.AddChunk(CodeCompletionString::CK_RightParen);
  Builder.AddChunk(CodeCompletionString::CK_HorizontalSpace);
  Builder.AddInformativeChunk("const");

  labelAndInsertText(*Builder.TakeString());
  EXPECT_EQ(Label, "Foo(p1, p2) const");
  EXPECT_EQ(InsertText, "Foo");
}

TEST_F(CompletionStringTest, FunctionSnippet) {
  Builder.AddResultTypeChunk("result no no");
  Builder.addBriefComment("This comment is ignored");
  Builder.AddTypedTextChunk("Foo");
  Builder.AddChunk(CodeCompletionString::CK_LeftParen);
  Builder.AddPlaceholderChunk("p1");
  Builder.AddChunk(CodeCompletionString::CK_Comma);
  Builder.AddPlaceholderChunk("p2");
  Builder.AddChunk(CodeCompletionString::CK_RightParen);

  auto *CCS = Builder.TakeString();
  labelAndInsertText(*CCS);
  EXPECT_EQ(Label, "Foo(p1, p2)");
  EXPECT_EQ(InsertText, "Foo");

  labelAndInsertText(*CCS, /*EnableSnippets=*/true);
  EXPECT_EQ(Label, "Foo(p1, p2)");
  EXPECT_EQ(InsertText, "Foo(${1:p1}, ${2:p2})");
  EXPECT_EQ(formatDocumentation(*CCS, "Foo's comment"), "Foo's comment");
}

TEST_F(CompletionStringTest, EscapeSnippet) {
  Builder.AddTypedTextChunk("Foo");
  Builder.AddChunk(CodeCompletionString::CK_LeftParen);
  Builder.AddPlaceholderChunk("$p}1\\");
  Builder.AddChunk(CodeCompletionString::CK_RightParen);

  labelAndInsertText(*Builder.TakeString(), /*EnableSnippets=*/true);
  EXPECT_EQ(Label, "Foo($p}1\\)");
  EXPECT_EQ(InsertText, "Foo(${1:\\$p\\}1\\\\})");
}

TEST_F(CompletionStringTest, IgnoreInformativeQualifier) {
  Builder.AddTypedTextChunk("X");
  Builder.AddInformativeChunk("info ok");
  Builder.AddInformativeChunk("info no no::");
  labelAndInsertText(*Builder.TakeString());
  EXPECT_EQ(Label, "Xinfo ok");
  EXPECT_EQ(InsertText, "X");
}

} // namespace
} // namespace clangd
} // namespace clang
