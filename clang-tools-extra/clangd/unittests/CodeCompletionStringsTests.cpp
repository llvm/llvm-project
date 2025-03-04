//===-- CodeCompletionStringsTests.cpp --------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CodeCompletionStrings.h"
#include "TestTU.h"
#include "clang/Sema/CodeCompleteConsumer.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <optional>

namespace clang {
namespace clangd {
namespace {

class CompletionStringTest : public ::testing::Test {
public:
  CompletionStringTest()
      : Allocator(std::make_shared<clang::GlobalCodeCompletionAllocator>()),
        CCTUInfo(Allocator), Builder(*Allocator, CCTUInfo) {}

protected:
  void computeSignature(const CodeCompletionString &CCS,
                        CodeCompletionResult::ResultKind ResultKind =
                            CodeCompletionResult::ResultKind::RK_Declaration,
                        bool IncludeFunctionArguments = true) {
    Signature.clear();
    Snippet.clear();
    getSignature(CCS, &Signature, &Snippet, ResultKind,
                 /*CursorKind=*/CXCursorKind::CXCursor_NotImplemented,
                 /*IncludeFunctionArguments=*/IncludeFunctionArguments,
                 /*RequiredQualifiers=*/nullptr);
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

TEST_F(CompletionStringTest, GetDeclCommentBadUTF8) {
  // <ff> is not a valid byte here, should be replaced by encoded <U+FFFD>.
  auto TU = TestTU::withCode("/*x\xffy*/ struct X;");
  auto AST = TU.build();
  EXPECT_EQ("x\xef\xbf\xbdy",
            getDeclComment(AST.getASTContext(), findDecl(AST, "X")));
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

TEST_F(CompletionStringTest, FunctionWithDefaultParams) {
  // return_type foo(p1, p2 = 0, p3 = 0)
  Builder.AddChunk(CodeCompletionString::CK_Comma);
  Builder.AddTypedTextChunk("p3 = 0");
  auto *DefaultParam2 = Builder.TakeString();

  Builder.AddChunk(CodeCompletionString::CK_Comma);
  Builder.AddTypedTextChunk("p2 = 0");
  Builder.AddOptionalChunk(DefaultParam2);
  auto *DefaultParam1 = Builder.TakeString();

  Builder.AddResultTypeChunk("return_type");
  Builder.AddTypedTextChunk("Foo");
  Builder.AddChunk(CodeCompletionString::CK_LeftParen);
  Builder.AddPlaceholderChunk("p1");
  Builder.AddOptionalChunk(DefaultParam1);
  Builder.AddChunk(CodeCompletionString::CK_RightParen);

  auto *CCS = Builder.TakeString();
  computeSignature(*CCS);
  EXPECT_EQ(Signature, "(p1, p2 = 0, p3 = 0)");
  EXPECT_EQ(Snippet, "(${1:p1})");
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

TEST_F(CompletionStringTest, SnippetsInPatterns) {
  auto MakeCCS = [this]() -> const CodeCompletionString & {
    CodeCompletionBuilder Builder(*Allocator, CCTUInfo);
    Builder.AddTypedTextChunk("namespace");
    Builder.AddChunk(CodeCompletionString::CK_HorizontalSpace);
    Builder.AddPlaceholderChunk("name");
    Builder.AddChunk(CodeCompletionString::CK_Equal);
    Builder.AddPlaceholderChunk("target");
    Builder.AddChunk(CodeCompletionString::CK_SemiColon);
    return *Builder.TakeString();
  };
  computeSignature(MakeCCS());
  EXPECT_EQ(Snippet, " ${1:name} = ${2:target};");

  // When completing a pattern, the last placeholder holds the cursor position.
  computeSignature(MakeCCS(),
                   /*ResultKind=*/CodeCompletionResult::ResultKind::RK_Pattern);
  EXPECT_EQ(Snippet, " ${1:name} = $0;");
}

TEST_F(CompletionStringTest, DropFunctionArguments) {
  Builder.AddTypedTextChunk("foo");
  Builder.AddChunk(CodeCompletionString::CK_LeftAngle);
  Builder.AddPlaceholderChunk("typename T");
  Builder.AddChunk(CodeCompletionString::CK_Comma);
  Builder.AddPlaceholderChunk("int U");
  Builder.AddChunk(CodeCompletionString::CK_RightAngle);
  Builder.AddChunk(CodeCompletionString::CK_LeftParen);
  Builder.AddPlaceholderChunk("arg1");
  Builder.AddChunk(CodeCompletionString::CK_Comma);
  Builder.AddPlaceholderChunk("arg2");
  Builder.AddChunk(CodeCompletionString::CK_RightParen);

  computeSignature(
      *Builder.TakeString(),
      /*ResultKind=*/CodeCompletionResult::ResultKind::RK_Declaration,
      /*IncludeFunctionArguments=*/false);
  // Arguments dropped from snippet, kept in signature.
  EXPECT_EQ(Signature, "<typename T, int U>(arg1, arg2)");
  EXPECT_EQ(Snippet, "<${1:typename T}, ${2:int U}>");
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

TEST(CompletionString, ParseDocumentation) {

  llvm::BumpPtrAllocator Allocator;
  CommentOptions CommentOpts;
  comments::CommandTraits Traits(Allocator, CommentOpts);

  struct Case {
    llvm::StringRef Documentation;
    std::optional<SymbolPrintedType> SymbolType;
    std::optional<SymbolPrintedType> SymbolReturnType;
    std::optional<std::vector<SymbolParam>> SymbolParameters;
    llvm::StringRef ExpectedRenderMarkdown;
    llvm::StringRef ExpectedRenderPlainText;
  } Cases[] = {
      {
          "foo bar",
          std::nullopt,
          std::nullopt,
          std::nullopt,
          "foo bar",
          "foo bar",
      },
      {
          "foo\nbar\n",
          std::nullopt,
          std::nullopt,
          std::nullopt,
          "foo bar",
          "foo bar",
      },
      {
          "foo\n\nbar\n",
          std::nullopt,
          std::nullopt,
          std::nullopt,
          "foo  \nbar",
          "foo\nbar",
      },
      {
          "foo \\p bar baz",
          std::nullopt,
          std::nullopt,
          std::nullopt,
          "foo `bar` baz",
          "foo bar baz",
      },
      {
          "foo \\e bar baz",
          std::nullopt,
          std::nullopt,
          std::nullopt,
          "foo \\*bar\\* baz",
          "foo *bar* baz",
      },
      {
          "foo \\b bar baz",
          std::nullopt,
          std::nullopt,
          std::nullopt,
          "foo \\*\\*bar\\*\\* baz",
          "foo **bar** baz",
      },
      {
          "foo \\ref bar baz",
          std::nullopt,
          std::nullopt,
          std::nullopt,
          "foo \\*\\*\\\\ref\\*\\* \\*bar\\* baz",
          "foo **\\ref** *bar* baz",
      },
      {
          "foo @ref bar baz",
          std::nullopt,
          std::nullopt,
          std::nullopt,
          "foo \\*\\*@ref\\*\\* \\*bar\\* baz",
          "foo **@ref** *bar* baz",
      },
      {
          "\\throw exception foo",
          std::nullopt,
          std::nullopt,
          std::nullopt,
          "\\*\\*\\\\throw\\*\\* \\*exception\\* foo",
          "**\\throw** *exception* foo",
      },
      {"@throws exception foo\n\n@note bar\n\n@warning baz\n\n@details "
       "qux\n\nfree standing paragraph",
       std::nullopt, std::nullopt, std::nullopt,
       "Warning:  \n- baz\n\n---\n"
       "Note:  \n- bar\n\n---\n"
       "\\*\\*@throws\\*\\* \\*exception\\* foo  \n"
       "\\*\\*@details\\*\\* qux  \n"
       "free standing paragraph",
       "Warning:\n- baz\n\n"
       "Note:\n- bar\n\n"
       "**@throws** *exception* foo\n"
       "**@details** qux\n"
       "free standing paragraph"},
      {"@throws exception foo\n\n@note bar\n\n@warning baz\n\n@note another "
       "note\n\n@warning another warning\n\n@details qux\n\nfree standing "
       "paragraph",
       std::nullopt, std::nullopt, std::nullopt,
       "Warnings:  \n- baz\n- another warning\n\n---\n"
       "Notes:  \n- bar\n- another note\n\n---\n"
       "\\*\\*@throws\\*\\* \\*exception\\* foo  \n"
       "\\*\\*@details\\*\\* qux  \n"
       "free standing paragraph",
       "Warnings:\n- baz\n- another warning\n\n"
       "Notes:\n- bar\n- another note\n\n"
       "**@throws** *exception* foo\n"
       "**@details** qux\n"
       "free standing paragraph"},
      {
          "",
          SymbolPrintedType("my_type", "type"),
          std::nullopt,
          std::nullopt,
          "Type: `my_type (aka type)`",
          "Type: my_type (aka type)",
      },
      {
          "",
          SymbolPrintedType("my_type", "type"),
          SymbolPrintedType("my_ret_type", "type"),
          std::nullopt,
          "→ `my_ret_type (aka type)`",
          "→ my_ret_type (aka type)",
      },
      {
          "\\return foo",
          SymbolPrintedType("my_type", "type"),
          SymbolPrintedType("my_ret_type", "type"),
          std::nullopt,
          "→ `my_ret_type (aka type)`: foo",
          "→ my_ret_type (aka type): foo",
      },
      {
          "\\returns foo",
          SymbolPrintedType("my_type", "type"),
          SymbolPrintedType("my_ret_type", "type"),
          std::nullopt,
          "→ `my_ret_type (aka type)`: foo",
          "→ my_ret_type (aka type): foo",
      },
      {
          "",
          std::nullopt,
          std::nullopt,
          std::vector<SymbolParam>{
              {SymbolPrintedType("my_type", "type"), "foo", "default"}},
          "Parameters:  \n- `my_type foo = default (aka type)`",
          "Parameters:\n- my_type foo = default (aka type)",
      },
      {
          "\\param foo bar",
          std::nullopt,
          std::nullopt,
          std::vector<SymbolParam>{
              {SymbolPrintedType("my_type", "type"), "foo", "default"}},
          "Parameters:  \n- `my_type foo = default (aka type)`: bar",
          "Parameters:\n- my_type foo = default (aka type): bar",
      },
      {
          "\\param foo bar\n\n\\param baz qux",
          std::nullopt,
          std::nullopt,
          std::vector<SymbolParam>{
              {SymbolPrintedType("my_type", "type"), "foo", "default"}},
          "Parameters:  \n- `my_type foo = default (aka type)`: bar",
          "Parameters:\n- my_type foo = default (aka type): bar",
      },
      {
          "\\brief This is a brief description\n\nlonger description "
          "paragraph\n\n\\note foo\n\n\\warning warning\n\njust another "
          "paragraph\\param foo bar\\return baz",
          SymbolPrintedType("my_type", "type"),
          SymbolPrintedType("my_ret_type", "type"),
          std::vector<SymbolParam>{
              {SymbolPrintedType("my_type", "type"), "foo", "default"}},
          "This is a brief description  \n\n"
          "---\n"
          "→ `my_ret_type (aka type)`: baz  \n"
          "Parameters:  \n"
          "- `my_type foo = default (aka type)`: bar\n\n"
          "Warning:  \n"
          "- warning\n\n"
          "---\n"
          "Note:  \n"
          "- foo\n\n"
          "---\n"
          "longer description paragraph  \n"
          "just another paragraph",

          R"(This is a brief description

→ my_ret_type (aka type): baz
Parameters:
- my_type foo = default (aka type): bar
Warning:
- warning

Note:
- foo

longer description paragraph
just another paragraph)",
      }};

  for (const auto &C : Cases) {
    markup::Document Doc;
    docCommentToMarkup(Doc, C.Documentation, Allocator, Traits, C.SymbolType,
                       C.SymbolReturnType, C.SymbolParameters);
    EXPECT_EQ(Doc.asPlainText(), C.ExpectedRenderPlainText);
    EXPECT_EQ(Doc.asMarkdown(), C.ExpectedRenderMarkdown);
  }
}

} // namespace
} // namespace clangd
} // namespace clang
