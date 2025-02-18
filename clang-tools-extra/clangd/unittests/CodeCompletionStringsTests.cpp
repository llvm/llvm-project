//===-- CodeCompletionStringsTests.cpp --------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CodeCompletionStrings.h"
#include "SymbolDocumentationMatchers.h"
#include "TestTU.h"
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

TEST_F(CompletionStringTest, GetDeclDocumentationBadUTF8) {
  // <ff> is not a valid byte here, should be replaced by encoded <U+FFFD>.
  const std::string Code = llvm::formatv(R"cpp(
  /// \brief {0}
  /// \details {0}
  /// \param {0} {0}
  /// \warning {0}
  /// \note {0}
  /// \return {0}
  struct X;
  )cpp",
                                         "x\xffy");

  auto TU = TestTU::withCode(Code);
  auto AST = TU.build();

  const std::string Utf8Replacement = "x\xef\xbf\xbdy";
  SymbolDocumentationOwned ExpectedDoc;
  ExpectedDoc.Brief = Utf8Replacement;
  ExpectedDoc.Returns = Utf8Replacement;
  ExpectedDoc.Parameters = {{Utf8Replacement, Utf8Replacement}};
  ExpectedDoc.Notes = {Utf8Replacement};
  ExpectedDoc.Warnings = {Utf8Replacement};
  ExpectedDoc.Description = {"\\details " + Utf8Replacement};
  ExpectedDoc.CommentText = llvm::formatv(R"(\brief {0}
\details {0}
\param {0} {0}
\warning {0}
\note {0}
\return {0})", Utf8Replacement);

  EXPECT_THAT(getDeclDocumentation(AST.getASTContext(), findDecl(AST, "X")),
              matchesDoc(ExpectedDoc));
}

TEST_F(CompletionStringTest, DoxygenParsing) {
  struct {
    const char *const Code;
    const std::function<void(SymbolDocumentationOwned &)> ExpectedBuilder;
  } Cases[] = {
      {R"cpp(
    // Hello world
    void foo();
    )cpp",
       [](SymbolDocumentationOwned &Doc) { Doc.Description = "Hello world"; }},
      {R"cpp(
    /*! 
     * \brief brief
     * \details details
     */
    void foo();
    )cpp",
       [](SymbolDocumentationOwned &Doc) {
         Doc.Brief = "brief";
         Doc.Description = "\\details details";
       }},
      {R"cpp(
    /** 
     * @brief brief
     * @details details
     * @see somewhere else
     */
    void foo();
    )cpp",
       [](SymbolDocumentationOwned &Doc) {
         Doc.Brief = "brief";
         Doc.Description = "@details details\n\n@see somewhere else";
       }},
      {R"cpp(
    /*! 
     * @brief brief
     * @details details
     * @param foo foodoc
     * @throws ball at hoop
     * @note note1
     * @warning warning1
     * @note note2
     * @warning warning2
     * @param bar bardoc
     * @return something
     */
    void foo();
    )cpp",
       [](SymbolDocumentationOwned &Doc) {
         Doc.Brief = "brief";
         Doc.Description = "@details details\n\n@throws ball at hoop";
         Doc.Parameters = {{"foo", "foodoc"}, {"bar", "bardoc"}};
         Doc.Warnings = {"warning1", "warning2"};
         Doc.Notes = {"note1", "note2"};
         Doc.Returns = "something";
       }},
      {R"cpp(
    /// @brief Here's \b bold \e italic and \p code
    int foo;
    )cpp",
       [](SymbolDocumentationOwned &Doc) {
         Doc.Brief = "Here's **bold** *italic* and `code`";
       }}};

  for (const auto &Case : Cases) {
    SCOPED_TRACE(Case.Code);

    auto TU = TestTU::withCode(Case.Code);
    auto AST = TU.build();
    auto &Ctx = AST.getASTContext();
    const auto &Decl = findDecl(AST, "foo");

    SymbolDocumentationOwned ExpectedDoc;
    ExpectedDoc.CommentText =
        getCompletionComment(Ctx, &Decl)
            ->getFormattedText(Ctx.getSourceManager(), Ctx.getDiagnostics());
    Case.ExpectedBuilder(ExpectedDoc);

    EXPECT_THAT(getDeclDocumentation(Ctx, Decl), matchesDoc(ExpectedDoc));
  }
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

} // namespace
} // namespace clangd
} // namespace clang
