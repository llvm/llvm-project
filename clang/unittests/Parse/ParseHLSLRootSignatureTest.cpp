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

#include "clang/Parse/ParseHLSLRootSignature.h"
#include "gtest/gtest.h"

using namespace clang;

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
  void SetNoDiag() {
    Satisfied = true;
    ExpectedDiagID = std::nullopt;
  }

  void SetExpected(unsigned DiagID) {
    Satisfied = false;
    ExpectedDiagID = DiagID;
  }

  bool IsSatisfied() { return Satisfied; }
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

  std::unique_ptr<Preprocessor> CreatePP(StringRef Source,
                                         TrivialModuleLoader &ModLoader) {
    std::unique_ptr<llvm::MemoryBuffer> Buf =
        llvm::MemoryBuffer::getMemBuffer(Source);
    SourceMgr.setMainFileID(SourceMgr.createFileID(std::move(Buf)));

    HeaderSearch HeaderInfo(std::make_shared<HeaderSearchOptions>(), SourceMgr,
                            Diags, LangOpts, Target.get());
    std::unique_ptr<Preprocessor> PP = std::make_unique<Preprocessor>(
        std::make_shared<PreprocessorOptions>(), Diags, LangOpts, SourceMgr,
        HeaderInfo, ModLoader,
        /*IILookup =*/nullptr,
        /*OwnsHeaderSearch =*/false);
    PP->Initialize(*Target);
    PP->EnterMainSourceFile();
    return PP;
  }

  void CheckTokens(hlsl::RootSignatureLexer &Lexer,
                   SmallVector<hlsl::RootSignatureToken> &Computed,
                   SmallVector<hlsl::TokenKind> &Expected) {
    for (unsigned I = 0, E = Expected.size(); I != E; ++I) {
      if (Expected[I] == hlsl::TokenKind::error ||
          Expected[I] == hlsl::TokenKind::invalid ||
          Expected[I] == hlsl::TokenKind::end_of_stream)
        continue;
      ASSERT_FALSE(Lexer.ConsumeToken());
      hlsl::RootSignatureToken Result = Lexer.GetCurToken();
      ASSERT_EQ(Result.Kind, Expected[I]);
      Computed.push_back(Result);
    }
    ASSERT_TRUE(Lexer.EndOfBuffer());
  }

  FileSystemOptions FileMgrOpts;
  FileManager FileMgr;
  IntrusiveRefCntPtr<DiagnosticIDs> DiagID;
  ExpectedDiagConsumer *Consumer;
  DiagnosticsEngine Diags;
  SourceManager SourceMgr;
  LangOptions LangOpts;
  std::shared_ptr<TargetOptions> TargetOpts;
  IntrusiveRefCntPtr<TargetInfo> Target;
};

// Valid Lexing Tests

TEST_F(ParseHLSLRootSignatureTest, ValidLexNumbersTest) {
  // This test will check that we can lex different number tokens
  const llvm::StringLiteral Source = R"cc(
    -42 42 +42 +2147483648
  )cc";

  TrivialModuleLoader ModLoader;
  auto PP = CreatePP(Source, ModLoader);
  auto TokLoc = SourceLocation();

  // Test no diagnostics produced
  Consumer->SetNoDiag();

  hlsl::RootSignatureLexer Lexer(Source, TokLoc, *PP);

  SmallVector<hlsl::RootSignatureToken> Tokens;
  SmallVector<hlsl::TokenKind> Expected = {
      hlsl::TokenKind::pu_minus,    hlsl::TokenKind::int_literal,
      hlsl::TokenKind::int_literal, hlsl::TokenKind::pu_plus,
      hlsl::TokenKind::int_literal, hlsl::TokenKind::pu_plus,
      hlsl::TokenKind::int_literal,
  };
  CheckTokens(Lexer, Tokens, Expected);
  ASSERT_TRUE(Consumer->IsSatisfied());

  // // Sample negative: int component
  hlsl::RootSignatureToken IntToken = Tokens[1];
  ASSERT_FALSE(IntToken.NumLiteral.getInt().isSigned());
  ASSERT_EQ(IntToken.NumLiteral.getInt().getExtValue(), 42);

  // Sample unsigned int
  IntToken = Tokens[2];
  ASSERT_FALSE(IntToken.NumLiteral.getInt().isSigned());
  ASSERT_EQ(IntToken.NumLiteral.getInt().getExtValue(), 42);

  // Sample positive: int component
  IntToken = Tokens[4];
  ASSERT_FALSE(IntToken.NumLiteral.getInt().isSigned());
  ASSERT_EQ(IntToken.NumLiteral.getInt().getExtValue(), 42);

  // Sample positive int that would overflow the signed representation but
  // is treated as an unsigned integer instead
  IntToken = Tokens[6];
  ASSERT_FALSE(IntToken.NumLiteral.getInt().isSigned());
  ASSERT_EQ(IntToken.NumLiteral.getInt().getExtValue(), 2147483648);
}

TEST_F(ParseHLSLRootSignatureTest, ValidLexAllTokensTest) {
  // This test will check that we can lex all defined tokens as defined in
  // HLSLRootSignatureTokenKinds.def, plus some additional integer variations
  const llvm::StringLiteral Source = R"cc(
    42

    b0 t43 u987 s234

    (),|=+-

    DescriptorTable

    CBV SRV UAV Sampler
    space visibility flags
    numDescriptors offset

    DESCRIPTOR_RANGE_OFFSET_APPEND

    DATA_VOLATILE
    DATA_STATIC_WHILE_SET_AT_EXECUTE
    DATA_STATIC
    DESCRIPTORS_VOLATILE
    DESCRIPTORS_STATIC_KEEPING_BUFFER_BOUNDS_CHECKS

    shader_visibility_all
    shader_visibility_vertex
    shader_visibility_hull
    shader_visibility_domain
    shader_visibility_geometry
    shader_visibility_pixel
    shader_visibility_amplification
    shader_visibility_mesh
  )cc";

  TrivialModuleLoader ModLoader;
  auto PP = CreatePP(Source, ModLoader);
  auto TokLoc = SourceLocation();

  // Test no diagnostics produced
  Consumer->SetNoDiag();

  hlsl::RootSignatureLexer Lexer(Source, TokLoc, *PP);

  SmallVector<hlsl::RootSignatureToken> Tokens;
  SmallVector<hlsl::TokenKind> Expected = {
#define TOK(NAME) hlsl::TokenKind::NAME,
#include "clang/Parse/HLSLRootSignatureTokenKinds.def"
  };

  CheckTokens(Lexer, Tokens, Expected);
  ASSERT_TRUE(Consumer->IsSatisfied());
}

TEST_F(ParseHLSLRootSignatureTest, ValidLexPeekTest) {
  // This test will check that we the peek api is correctly used
  const llvm::StringLiteral Source = R"cc(
    )1
  )cc";

  TrivialModuleLoader ModLoader;
  auto PP = CreatePP(Source, ModLoader);
  auto TokLoc = SourceLocation();

  // Test no diagnostics produced
  Consumer->SetNoDiag();

  hlsl::RootSignatureLexer Lexer(Source, TokLoc, *PP);
  // Test basic peek
  auto Res = Lexer.PeekNextToken();
  ASSERT_TRUE(Res.has_value());
  ASSERT_EQ(Res->Kind, hlsl::TokenKind::pu_r_paren);

  // Ensure it doesn't peek past one element
  Res = Lexer.PeekNextToken();
  ASSERT_TRUE(Res.has_value());
  ASSERT_EQ(Res->Kind, hlsl::TokenKind::pu_r_paren);

  ASSERT_FALSE(Lexer.ConsumeToken());

  // Invoke after reseting the NextToken
  Res = Lexer.PeekNextToken();
  ASSERT_TRUE(Res.has_value());
  ASSERT_EQ(Res->Kind, hlsl::TokenKind::int_literal);

  // Ensure we can still consume the second token
  ASSERT_FALSE(Lexer.ConsumeToken());

  // Ensure no error raised when peeking past end of stream
  Res = Lexer.PeekNextToken();
  ASSERT_TRUE(Res.has_value());
  ASSERT_EQ(Res->Kind, hlsl::TokenKind::end_of_stream);

  // Ensure no error raised when consuming past end of stream
  ASSERT_FALSE(Lexer.ConsumeToken());
  ASSERT_EQ(Lexer.GetCurToken().Kind, hlsl::TokenKind::end_of_stream);

  ASSERT_TRUE(Consumer->IsSatisfied());
}

// Invalid Lexing Tests

TEST_F(ParseHLSLRootSignatureTest, InvalidLexOverflowedNumberTest) {
  // This test will check that the lexing fails due to an integer overflow
  const llvm::StringLiteral Source = R"cc(
    4294967296
  )cc";

  TrivialModuleLoader ModLoader;
  auto PP = CreatePP(Source, ModLoader);
  auto TokLoc = SourceLocation();

  // Test correct diagnostic produced
  Consumer->SetExpected(diag::err_hlsl_number_literal_overflow);

  hlsl::RootSignatureLexer Lexer(Source, TokLoc, *PP);

  // We will also test that the error can be produced during peek and the Lexer
  // will correctly just return true on the next ConsumeToken without
  // reporting another error
  auto Result = Lexer.PeekNextToken();
  ASSERT_EQ(std::nullopt, Result);
  ASSERT_TRUE(Lexer.ConsumeToken());
  ASSERT_TRUE(Consumer->IsSatisfied());
}

TEST_F(ParseHLSLRootSignatureTest, InvalidLexRegNumberTest) {
  // This test will check that the lexing fails due to no integer being provided
  const llvm::StringLiteral Source = R"cc(
    b32.4
  )cc";

  TrivialModuleLoader ModLoader;
  auto PP = CreatePP(Source, ModLoader);
  auto TokLoc = SourceLocation();

  // Test correct diagnostic produced
  Consumer->SetExpected(diag::err_hlsl_invalid_register_literal);

  hlsl::RootSignatureLexer Lexer(Source, TokLoc, *PP);

  ASSERT_TRUE(Lexer.ConsumeToken());
  // FIXME(#126565): This should be TRUE once we can lex a float
  ASSERT_FALSE(Consumer->IsSatisfied());
}

} // anonymous namespace
