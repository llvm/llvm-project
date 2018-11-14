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
  void computeSignature(const CodeCompletionString &CCS) {
    Signature.clear();
    Snippet.clear();
    getSignature(CCS, &Signature, &Snippet);
  }

  std::shared_ptr<clang::GlobalCodeCompletionAllocator> Allocator;
  CodeCompletionTUInfo CCTUInfo;
  CodeCompletionBuilder Builder;
  std::string Signature;
  std::string Snippet;
};

TEST_F(CompletionStringTest, ReturnType) {
  Builder.AddResultTypeChunk("result");
  Builder.AddResultTypeChunk("redundant result no no");
  EXPECT_EQ(getReturnType(*Builder.TakeString()), "result");
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

TEST_F(CompletionStringTest, EmptySignature) {
  Builder.AddTypedTextChunk("X");
  Builder.AddResultTypeChunk("result no no");
  computeSignature(*Builder.TakeString());
  EXPECT_EQ(Signature, "");
  EXPECT_EQ(Snippet, "");
}

TEST_F(CompletionStringTest, Function) {
  Builder.AddResultTypeChunk("result no no");
  Builder.addBriefComment("This comment is ignored");
  Builder.AddTypedTextChunk("Foo");
  Builder.AddChunk(CodeCompletionString::CK_LeftParen);
  Builder.AddPlaceholderChunk("p1");
  Builder.AddChunk(CodeCompletionString::CK_Comma);
  Builder.AddPlaceholderChunk("p2");
  Builder.AddChunk(CodeCompletionString::CK_RightParen);

  auto *CCS = Builder.TakeString();
  computeSignature(*CCS);
  EXPECT_EQ(Signature, "(p1, p2)");
  EXPECT_EQ(Snippet, "(${1:p1}, ${2:p2})");
  EXPECT_EQ(formatDocumentation(*CCS, "Foo's comment"), "Foo's comment");
}

TEST_F(CompletionStringTest, EscapeSnippet) {
  Builder.AddTypedTextChunk("Foo");
  Builder.AddChunk(CodeCompletionString::CK_LeftParen);
  Builder.AddPlaceholderChunk("$p}1\\");
  Builder.AddChunk(CodeCompletionString::CK_RightParen);

  computeSignature(*Builder.TakeString());
  EXPECT_EQ(Signature, "($p}1\\)");
  EXPECT_EQ(Snippet, "(${1:\\$p\\}1\\\\})");
}

TEST_F(CompletionStringTest, IgnoreInformativeQualifier) {
  Builder.AddTypedTextChunk("X");
  Builder.AddInformativeChunk("info ok");
  Builder.AddInformativeChunk("info no no::");
  computeSignature(*Builder.TakeString());
  EXPECT_EQ(Signature, "info ok");
  EXPECT_EQ(Snippet, "");
}

TEST_F(CompletionStringTest, ObjectiveCMethodNoArguments) {
  Builder.AddResultTypeChunk("void");
  Builder.AddTypedTextChunk("methodName");

  auto *CCS = Builder.TakeString();
  computeSignature(*CCS);
  EXPECT_EQ(Signature, "");
  EXPECT_EQ(Snippet, "");
}

TEST_F(CompletionStringTest, ObjectiveCMethodOneArgument) {
  Builder.AddResultTypeChunk("void");
  Builder.AddTypedTextChunk("methodWithArg:");
  Builder.AddPlaceholderChunk("(type)");

  auto *CCS = Builder.TakeString();
  computeSignature(*CCS);
  EXPECT_EQ(Signature, "(type)");
  EXPECT_EQ(Snippet, "${1:(type)}");
}

TEST_F(CompletionStringTest, ObjectiveCMethodTwoArgumentsFromBeginning) {
  Builder.AddResultTypeChunk("int");
  Builder.AddTypedTextChunk("withFoo:");
  Builder.AddPlaceholderChunk("(type)");
  Builder.AddChunk(CodeCompletionString::CK_HorizontalSpace);
  Builder.AddTypedTextChunk("bar:");
  Builder.AddPlaceholderChunk("(type2)");

  auto *CCS = Builder.TakeString();
  computeSignature(*CCS);
  EXPECT_EQ(Signature, "(type) bar:(type2)");
  EXPECT_EQ(Snippet, "${1:(type)} bar:${2:(type2)}");
}

TEST_F(CompletionStringTest, ObjectiveCMethodTwoArgumentsFromMiddle) {
  Builder.AddResultTypeChunk("int");
  Builder.AddInformativeChunk("withFoo:");
  Builder.AddTypedTextChunk("bar:");
  Builder.AddPlaceholderChunk("(type2)");

  auto *CCS = Builder.TakeString();
  computeSignature(*CCS);
  EXPECT_EQ(Signature, "(type2)");
  EXPECT_EQ(Snippet, "${1:(type2)}");
}

} // namespace
} // namespace clangd
} // namespace clang
