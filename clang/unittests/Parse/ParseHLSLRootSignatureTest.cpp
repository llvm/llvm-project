//=== ParseHLSLRootSignatureTest.cpp - Parse Root Signature tests ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/DiagnosticOptions.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/LangOptions.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Lex/HeaderSearch.h"
#include "clang/Lex/HeaderSearchOptions.h"
#include "clang/Lex/Lexer.h"
#include "clang/Lex/ModuleLoader.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Lex/PreprocessorOptions.h"

#include "clang/Lex/LexHLSLRootSignature.h"
#include "clang/Parse/ParseHLSLRootSignature.h"
#include "gtest/gtest.h"

using namespace clang;
using namespace llvm::hlsl::rootsig;

namespace {

// Diagnostic helper for helper tests
class ExpectedDiagConsumer : public DiagnosticConsumer {
  virtual void anchor() {}

  void HandleDiagnostic(DiagnosticsEngine::Level DiagLevel,
                        const Diagnostic &Info) override {
    if (!FirstDiag || !ExpectedDiagID.has_value()) {
      Satisfied = false;
      return;
    }
    FirstDiag = false;

    Satisfied = ExpectedDiagID.value() == Info.getID();
  }

  bool FirstDiag = true;
  bool Satisfied = false;
  std::optional<unsigned> ExpectedDiagID;

public:
  void setNoDiag() {
    Satisfied = true;
    ExpectedDiagID = std::nullopt;
  }

  void setExpected(unsigned DiagID) {
    Satisfied = false;
    ExpectedDiagID = DiagID;
  }

  bool isSatisfied() { return Satisfied; }
};

// The test fixture.
class ParseHLSLRootSignatureTest : public ::testing::Test {
protected:
  ParseHLSLRootSignatureTest()
      : FileMgr(FileMgrOpts), DiagID(new DiagnosticIDs()),
        Consumer(new ExpectedDiagConsumer()),
        Diags(DiagID, new DiagnosticOptions, Consumer),
        SourceMgr(Diags, FileMgr), TargetOpts(new TargetOptions) {
    // This is an arbitrarily chosen target triple to create the target info.
    TargetOpts->Triple = "dxil";
    Target = TargetInfo::CreateTargetInfo(Diags, TargetOpts);
  }

  std::unique_ptr<Preprocessor> createPP(StringRef Source,
                                         TrivialModuleLoader &ModLoader) {
    std::unique_ptr<llvm::MemoryBuffer> Buf =
        llvm::MemoryBuffer::getMemBuffer(Source);
    SourceMgr.setMainFileID(SourceMgr.createFileID(std::move(Buf)));

    HeaderSearchOptions SearchOpts;
    HeaderSearch HeaderInfo(SearchOpts, SourceMgr, Diags, LangOpts,
                            Target.get());
    auto PP = std::make_unique<Preprocessor>(
        PPOpts, Diags, LangOpts, SourceMgr, HeaderInfo, ModLoader,
        /*IILookup =*/nullptr, /*OwnsHeaderSearch =*/false);
    PP->Initialize(*Target);
    PP->EnterMainSourceFile();
    return PP;
  }

  FileSystemOptions FileMgrOpts;
  FileManager FileMgr;
  IntrusiveRefCntPtr<DiagnosticIDs> DiagID;
  ExpectedDiagConsumer *Consumer;
  DiagnosticsEngine Diags;
  SourceManager SourceMgr;
  LangOptions LangOpts;
  PreprocessorOptions PPOpts;
  std::shared_ptr<TargetOptions> TargetOpts;
  IntrusiveRefCntPtr<TargetInfo> Target;
};

// Valid Parser Tests

TEST_F(ParseHLSLRootSignatureTest, ValidParseEmptyTest) {
  const llvm::StringLiteral Source = R"cc()cc";

  TrivialModuleLoader ModLoader;
  auto PP = createPP(Source, ModLoader);
  auto TokLoc = SourceLocation();

  hlsl::RootSignatureLexer Lexer(Source, TokLoc);
  SmallVector<RootElement> Elements;
  hlsl::RootSignatureParser Parser(Elements, Lexer, *PP);

  // Test no diagnostics produced
  Consumer->setNoDiag();

  ASSERT_FALSE(Parser.parse());
  ASSERT_EQ((int)Elements.size(), 0);

  ASSERT_TRUE(Consumer->isSatisfied());
}

TEST_F(ParseHLSLRootSignatureTest, ValidParseDTClausesTest) {
  const llvm::StringLiteral Source = R"cc(
    DescriptorTable(
      CBV(b0),
      SRV(space = 3, t42),
      Sampler(s987, space = +2),
      UAV(u4294967294)
    ),
    DescriptorTable()
  )cc";

  TrivialModuleLoader ModLoader;
  auto PP = createPP(Source, ModLoader);
  auto TokLoc = SourceLocation();

  hlsl::RootSignatureLexer Lexer(Source, TokLoc);
  SmallVector<RootElement> Elements;
  hlsl::RootSignatureParser Parser(Elements, Lexer, *PP);

  // Test no diagnostics produced
  Consumer->setNoDiag();

  ASSERT_FALSE(Parser.parse());

  // First Descriptor Table with 4 elements
  RootElement Elem = Elements[0];
  ASSERT_TRUE(std::holds_alternative<DescriptorTableClause>(Elem));
  ASSERT_EQ(std::get<DescriptorTableClause>(Elem).Type, ClauseType::CBuffer);
  ASSERT_EQ(std::get<DescriptorTableClause>(Elem).Register.ViewType,
            RegisterType::BReg);
  ASSERT_EQ(std::get<DescriptorTableClause>(Elem).Register.Number, 0u);
  ASSERT_EQ(std::get<DescriptorTableClause>(Elem).Space, 0u);

  Elem = Elements[1];
  ASSERT_TRUE(std::holds_alternative<DescriptorTableClause>(Elem));
  ASSERT_EQ(std::get<DescriptorTableClause>(Elem).Type, ClauseType::SRV);
  ASSERT_EQ(std::get<DescriptorTableClause>(Elem).Register.ViewType,
            RegisterType::TReg);
  ASSERT_EQ(std::get<DescriptorTableClause>(Elem).Register.Number, 42u);
  ASSERT_EQ(std::get<DescriptorTableClause>(Elem).Space, 3u);

  Elem = Elements[2];
  ASSERT_TRUE(std::holds_alternative<DescriptorTableClause>(Elem));
  ASSERT_EQ(std::get<DescriptorTableClause>(Elem).Type, ClauseType::Sampler);
  ASSERT_EQ(std::get<DescriptorTableClause>(Elem).Register.ViewType,
            RegisterType::SReg);
  ASSERT_EQ(std::get<DescriptorTableClause>(Elem).Register.Number, 987u);
  ASSERT_EQ(std::get<DescriptorTableClause>(Elem).Space, 2u);

  Elem = Elements[3];
  ASSERT_TRUE(std::holds_alternative<DescriptorTableClause>(Elem));
  ASSERT_EQ(std::get<DescriptorTableClause>(Elem).Type, ClauseType::UAV);
  ASSERT_EQ(std::get<DescriptorTableClause>(Elem).Register.ViewType,
            RegisterType::UReg);
  ASSERT_EQ(std::get<DescriptorTableClause>(Elem).Register.Number, 4294967294u);
  ASSERT_EQ(std::get<DescriptorTableClause>(Elem).Space, 0u);

  Elem = Elements[4];
  ASSERT_TRUE(std::holds_alternative<DescriptorTable>(Elem));
  ASSERT_EQ(std::get<DescriptorTable>(Elem).NumClauses, (uint32_t)4);

  // Empty Descriptor Table
  Elem = Elements[5];
  ASSERT_TRUE(std::holds_alternative<DescriptorTable>(Elem));
  ASSERT_EQ(std::get<DescriptorTable>(Elem).NumClauses, 0u);

  ASSERT_TRUE(Consumer->isSatisfied());
}

TEST_F(ParseHLSLRootSignatureTest, ValidTrailingCommaTest) {
  // This test will checks we can handling trailing commas ','
  const llvm::StringLiteral Source = R"cc(
    DescriptorTable(
      CBV(b0, ),
      SRV(t42),
    )
  )cc";

  TrivialModuleLoader ModLoader;
  auto PP = createPP(Source, ModLoader);
  auto TokLoc = SourceLocation();

  hlsl::RootSignatureLexer Lexer(Source, TokLoc);
  SmallVector<RootElement> Elements;
  hlsl::RootSignatureParser Parser(Elements, Lexer, *PP);

  // Test no diagnostics produced
  Consumer->setNoDiag();

  ASSERT_FALSE(Parser.parse());

  ASSERT_TRUE(Consumer->isSatisfied());
}

// Invalid Parser Tests

TEST_F(ParseHLSLRootSignatureTest, InvalidParseUnexpectedTokenTest) {
  const llvm::StringLiteral Source = R"cc(
    DescriptorTable()
    space
  )cc";

  TrivialModuleLoader ModLoader;
  auto PP = createPP(Source, ModLoader);
  auto TokLoc = SourceLocation();

  hlsl::RootSignatureLexer Lexer(Source, TokLoc);
  SmallVector<RootElement> Elements;
  hlsl::RootSignatureParser Parser(Elements, Lexer, *PP);

  // Test correct diagnostic produced
  Consumer->setExpected(diag::err_hlsl_unexpected_end_of_params);
  ASSERT_TRUE(Parser.parse());

  ASSERT_TRUE(Consumer->isSatisfied());
}

TEST_F(ParseHLSLRootSignatureTest, InvalidParseInvalidTokenTest) {
  const llvm::StringLiteral Source = R"cc(
    notAnIdentifier
  )cc";

  TrivialModuleLoader ModLoader;
  auto PP = createPP(Source, ModLoader);
  auto TokLoc = SourceLocation();

  hlsl::RootSignatureLexer Lexer(Source, TokLoc);
  SmallVector<RootElement> Elements;
  hlsl::RootSignatureParser Parser(Elements, Lexer, *PP);

  // Test correct diagnostic produced - invalid token
  Consumer->setExpected(diag::err_hlsl_unexpected_end_of_params);
  ASSERT_TRUE(Parser.parse());

  ASSERT_TRUE(Consumer->isSatisfied());
}

TEST_F(ParseHLSLRootSignatureTest, InvalidParseUnexpectedEndOfStreamTest) {
  const llvm::StringLiteral Source = R"cc(
    DescriptorTable
  )cc";

  TrivialModuleLoader ModLoader;
  auto PP = createPP(Source, ModLoader);
  auto TokLoc = SourceLocation();

  hlsl::RootSignatureLexer Lexer(Source, TokLoc);
  SmallVector<RootElement> Elements;
  hlsl::RootSignatureParser Parser(Elements, Lexer, *PP);

  // Test correct diagnostic produced - end of stream
  Consumer->setExpected(diag::err_expected_after);

  ASSERT_TRUE(Parser.parse());

  ASSERT_TRUE(Consumer->isSatisfied());
}

TEST_F(ParseHLSLRootSignatureTest, InvalidMissingParameterTest) {
  // This test will check that the parsing fails due a mandatory
  // parameter (register) not being specified
  const llvm::StringLiteral Source = R"cc(
    DescriptorTable(
      CBV()
    )
  )cc";

  TrivialModuleLoader ModLoader;
  auto PP = createPP(Source, ModLoader);
  auto TokLoc = SourceLocation();

  hlsl::RootSignatureLexer Lexer(Source, TokLoc);
  SmallVector<RootElement> Elements;
  hlsl::RootSignatureParser Parser(Elements, Lexer, *PP);

  // Test correct diagnostic produced
  Consumer->setExpected(diag::err_hlsl_rootsig_missing_param);
  ASSERT_TRUE(Parser.parse());

  ASSERT_TRUE(Consumer->isSatisfied());
}

TEST_F(ParseHLSLRootSignatureTest, InvalidRepeatedMandatoryParameterTest) {
  // This test will check that the parsing fails due the same mandatory
  // parameter being specified multiple times
  const llvm::StringLiteral Source = R"cc(
    DescriptorTable(
      CBV(b32, b84)
    )
  )cc";

  TrivialModuleLoader ModLoader;
  auto PP = createPP(Source, ModLoader);
  auto TokLoc = SourceLocation();

  hlsl::RootSignatureLexer Lexer(Source, TokLoc);
  SmallVector<RootElement> Elements;
  hlsl::RootSignatureParser Parser(Elements, Lexer, *PP);

  // Test correct diagnostic produced
  Consumer->setExpected(diag::err_hlsl_rootsig_repeat_param);
  ASSERT_TRUE(Parser.parse());

  ASSERT_TRUE(Consumer->isSatisfied());
}

TEST_F(ParseHLSLRootSignatureTest, InvalidRepeatedOptionalParameterTest) {
  // This test will check that the parsing fails due the same optional
  // parameter being specified multiple times
  const llvm::StringLiteral Source = R"cc(
    DescriptorTable(
      CBV(space = 2, space = 0)
    )
  )cc";

  TrivialModuleLoader ModLoader;
  auto PP = createPP(Source, ModLoader);
  auto TokLoc = SourceLocation();

  hlsl::RootSignatureLexer Lexer(Source, TokLoc);
  SmallVector<RootElement> Elements;
  hlsl::RootSignatureParser Parser(Elements, Lexer, *PP);

  // Test correct diagnostic produced
  Consumer->setExpected(diag::err_hlsl_rootsig_repeat_param);
  ASSERT_TRUE(Parser.parse());

  ASSERT_TRUE(Consumer->isSatisfied());
}

TEST_F(ParseHLSLRootSignatureTest, InvalidLexOverflowedNumberTest) {
  // This test will check that the lexing fails due to an integer overflow
  const llvm::StringLiteral Source = R"cc(
    DescriptorTable(
      CBV(b4294967296)
    )
  )cc";

  TrivialModuleLoader ModLoader;
  auto PP = createPP(Source, ModLoader);
  auto TokLoc = SourceLocation();

  hlsl::RootSignatureLexer Lexer(Source, TokLoc);
  SmallVector<RootElement> Elements;
  hlsl::RootSignatureParser Parser(Elements, Lexer, *PP);

  // Test correct diagnostic produced
  Consumer->setExpected(diag::err_hlsl_number_literal_overflow);
  ASSERT_TRUE(Parser.parse());

  ASSERT_TRUE(Consumer->isSatisfied());
}

} // anonymous namespace
