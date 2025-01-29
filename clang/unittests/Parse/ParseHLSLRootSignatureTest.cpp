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
using namespace llvm::hlsl::root_signature;

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
    TargetOpts->Triple = "x86_64-apple-darwin11.1.0";
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

  void CheckTokens(SmallVector<hlsl::RootSignatureToken> &Computed,
                   SmallVector<hlsl::TokenKind> &Expected) {
    ASSERT_EQ(Computed.size(), Expected.size());
    for (unsigned I = 0, E = Expected.size(); I != E; ++I) {
      ASSERT_EQ(Computed[I].Kind, Expected[I]);
    }
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
    -42 42 +42
  )cc";

  TrivialModuleLoader ModLoader;
  auto PP = CreatePP(Source, ModLoader);
  auto TokLoc = SourceLocation();

  // Test no diagnostics produced
  Consumer->SetNoDiag();

  hlsl::RootSignatureLexer Lexer(Source, TokLoc, *PP);

  SmallVector<hlsl::RootSignatureToken> Tokens;
  ASSERT_FALSE(Lexer.Lex(Tokens));
  ASSERT_TRUE(Consumer->IsSatisfied());

  SmallVector<hlsl::TokenKind> Expected = {
      hlsl::TokenKind::int_literal,
      hlsl::TokenKind::int_literal,
      hlsl::TokenKind::int_literal,
  };
  CheckTokens(Tokens, Expected);
}

TEST_F(ParseHLSLRootSignatureTest, ValidLexAllTokensTest) {
  // This test will check that we can lex all defined tokens as defined in
  // HLSLRootSignatureTokenKinds.def, plus some additional integer variations
  const llvm::StringLiteral Source = R"cc(
    42

    b0 t43 u987 s234

    (),|=

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

  SmallVector<hlsl::RootSignatureToken> Tokens = {
      hlsl::RootSignatureToken(
          SourceLocation()) // invalid token for completeness
  };
  ASSERT_FALSE(Lexer.Lex(Tokens));
  ASSERT_TRUE(Consumer->IsSatisfied());

  SmallVector<hlsl::TokenKind> Expected = {
#define TOK(NAME) hlsl::TokenKind::NAME,
#include "clang/Parse/HLSLRootSignatureTokenKinds.def"
  };

  CheckTokens(Tokens, Expected);
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

  SmallVector<hlsl::RootSignatureToken> Tokens;
  ASSERT_TRUE(Lexer.Lex(Tokens));
  ASSERT_TRUE(Consumer->IsSatisfied());
}

TEST_F(ParseHLSLRootSignatureTest, InvalidLexEmptyNumberTest) {
  // This test will check that the lexing fails due to no integer being provided
  const llvm::StringLiteral Source = R"cc(
    -
  )cc";

  TrivialModuleLoader ModLoader;
  auto PP = CreatePP(Source, ModLoader);
  auto TokLoc = SourceLocation();

  // Test correct diagnostic produced
  Consumer->SetExpected(diag::err_hlsl_invalid_number_literal);

  hlsl::RootSignatureLexer Lexer(Source, TokLoc, *PP);

  SmallVector<hlsl::RootSignatureToken> Tokens;
  ASSERT_TRUE(Lexer.Lex(Tokens));
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

  SmallVector<hlsl::RootSignatureToken> Tokens;
  ASSERT_TRUE(Lexer.Lex(Tokens));
  // FIXME(#120472): This should be TRUE once we can lex a floating
  ASSERT_FALSE(Consumer->IsSatisfied());
}

TEST_F(ParseHLSLRootSignatureTest, InvalidLexIdentifierTest) {
  // This test will check that the lexing fails due to no valid token
  const llvm::StringLiteral Source = R"cc(
    notAToken
  )cc";

  TrivialModuleLoader ModLoader;
  auto PP = CreatePP(Source, ModLoader);
  auto TokLoc = SourceLocation();

  // Test correct diagnostic produced
  Consumer->SetExpected(diag::err_hlsl_invalid_token);

  hlsl::RootSignatureLexer Lexer(Source, TokLoc, *PP);

  SmallVector<hlsl::RootSignatureToken> Tokens;
  ASSERT_TRUE(Lexer.Lex(Tokens));
  ASSERT_TRUE(Consumer->IsSatisfied());
}

// Valid Parser Tests

TEST_F(ParseHLSLRootSignatureTest, ValidParseEmptyTest) {
  const llvm::StringLiteral Source = R"cc()cc";

  TrivialModuleLoader ModLoader;
  auto PP = CreatePP(Source, ModLoader);
  auto TokLoc = SourceLocation();

  // Test no diagnostics produced
  Consumer->SetNoDiag();
  hlsl::RootSignatureLexer Lexer(Source, TokLoc, *PP);

  SmallVector<hlsl::RootSignatureToken> Tokens;
  ASSERT_FALSE(Lexer.Lex(Tokens));

  SmallVector<RootElement> Elements;
  hlsl::RootSignatureParser Parser(Elements, Tokens, Diags);

  ASSERT_FALSE(Parser.Parse());
  ASSERT_EQ((int)Elements.size(), 0);

  ASSERT_TRUE(Consumer->IsSatisfied());
}

TEST_F(ParseHLSLRootSignatureTest, ValidParseDTClausesTest) {
  const llvm::StringLiteral Source = R"cc(
    DescriptorTable(
      visibility = SHADER_VISIBILITY_PIXEL,
      CBV(b0),
      SRV(t42, space = 3, offset = 32, numDescriptors = 4, flags = 0),
      Sampler(s987, space = 2, offset = DESCRIPTOR_RANGE_OFFSET_APPEND),
      UAV(u987234,
        flags = Descriptors_Volatile | Data_Volatile
                      | Data_Static_While_Set_At_Execute | Data_Static
                      | Descriptors_Static_Keeping_Buffer_Bounds_Checks
      )
    ),
    DescriptorTable()
  )cc";

  TrivialModuleLoader ModLoader;
  auto PP = CreatePP(Source, ModLoader);
  auto TokLoc = SourceLocation();

  hlsl::RootSignatureLexer Lexer(Source, TokLoc, *PP);

  SmallVector<hlsl::RootSignatureToken> Tokens;

  // Test no diagnostics produced
  Consumer->SetNoDiag();
  ASSERT_FALSE(Lexer.Lex(Tokens));

  SmallVector<RootElement> Elements;
  hlsl::RootSignatureParser Parser(Elements, Tokens, Diags);

  ASSERT_FALSE(Parser.Parse());
  RootElement Elem = Elements[0];
  ASSERT_TRUE(std::holds_alternative<DescriptorTableClause>(Elem));
  ASSERT_EQ(std::get<DescriptorTableClause>(Elem).Type, ClauseType::CBuffer);
  ASSERT_EQ(std::get<DescriptorTableClause>(Elem).Register.ViewType,
            RegisterType::BReg);
  ASSERT_EQ(std::get<DescriptorTableClause>(Elem).Register.Number, (uint32_t)0);
  ASSERT_EQ(std::get<DescriptorTableClause>(Elem).NumDescriptors, (uint32_t)1);
  ASSERT_EQ(std::get<DescriptorTableClause>(Elem).Space, (uint32_t)0);
  ASSERT_EQ(std::get<DescriptorTableClause>(Elem).Offset,
            DescriptorRangeOffset(DescriptorTableOffsetAppend));
  ASSERT_EQ(std::get<DescriptorTableClause>(Elem).Flags,
            DescriptorRangeFlags::DataStaticWhileSetAtExecute);

  Elem = Elements[1];
  ASSERT_TRUE(std::holds_alternative<DescriptorTableClause>(Elem));
  ASSERT_EQ(std::get<DescriptorTableClause>(Elem).Type, ClauseType::SRV);
  ASSERT_EQ(std::get<DescriptorTableClause>(Elem).Register.ViewType,
            RegisterType::TReg);
  ASSERT_EQ(std::get<DescriptorTableClause>(Elem).Register.Number,
            (uint32_t)42);
  ASSERT_EQ(std::get<DescriptorTableClause>(Elem).NumDescriptors, (uint32_t)4);
  ASSERT_EQ(std::get<DescriptorTableClause>(Elem).Space, (uint32_t)3);
  ASSERT_EQ(std::get<DescriptorTableClause>(Elem).Offset,
            DescriptorRangeOffset(32));
  ASSERT_EQ(std::get<DescriptorTableClause>(Elem).Flags,
            DescriptorRangeFlags::None);

  Elem = Elements[2];
  ASSERT_TRUE(std::holds_alternative<DescriptorTableClause>(Elem));
  ASSERT_EQ(std::get<DescriptorTableClause>(Elem).Type, ClauseType::Sampler);
  ASSERT_EQ(std::get<DescriptorTableClause>(Elem).Register.ViewType,
            RegisterType::SReg);
  ASSERT_EQ(std::get<DescriptorTableClause>(Elem).Register.Number,
            (uint32_t)987);
  ASSERT_EQ(std::get<DescriptorTableClause>(Elem).NumDescriptors, (uint32_t)1);
  ASSERT_EQ(std::get<DescriptorTableClause>(Elem).Space, (uint32_t)2);
  ASSERT_EQ(std::get<DescriptorTableClause>(Elem).Offset,
            DescriptorRangeOffset(DescriptorTableOffsetAppend));
  ASSERT_EQ(std::get<DescriptorTableClause>(Elem).Flags,
            DescriptorRangeFlags::None);

  Elem = Elements[3];
  ASSERT_TRUE(std::holds_alternative<DescriptorTableClause>(Elem));
  ASSERT_EQ(std::get<DescriptorTableClause>(Elem).Type, ClauseType::UAV);
  ASSERT_EQ(std::get<DescriptorTableClause>(Elem).Register.ViewType,
            RegisterType::UReg);
  ASSERT_EQ(std::get<DescriptorTableClause>(Elem).Register.Number,
            (uint32_t)987234);
  ASSERT_EQ(std::get<DescriptorTableClause>(Elem).NumDescriptors, (uint32_t)1);
  ASSERT_EQ(std::get<DescriptorTableClause>(Elem).Space, (uint32_t)0);
  ASSERT_EQ(std::get<DescriptorTableClause>(Elem).Offset,
            DescriptorRangeOffset(DescriptorTableOffsetAppend));
  ASSERT_EQ(std::get<DescriptorTableClause>(Elem).Flags,
            DescriptorRangeFlags::ValidFlags);

  Elem = Elements[4];
  ASSERT_TRUE(std::holds_alternative<DescriptorTable>(Elem));
  ASSERT_EQ(std::get<DescriptorTable>(Elem).NumClauses, (uint32_t)4);
  ASSERT_EQ(std::get<DescriptorTable>(Elem).Visibility,
            ShaderVisibility::Pixel);

  Elem = Elements[5];
  // Test generated DescriptorTable start has correct default values
  ASSERT_TRUE(std::holds_alternative<DescriptorTable>(Elem));
  ASSERT_EQ(std::get<DescriptorTable>(Elem).NumClauses, (uint32_t)0);
  ASSERT_EQ(std::get<DescriptorTable>(Elem).Visibility, ShaderVisibility::All);

  ASSERT_TRUE(Consumer->IsSatisfied());
}

// Invalid Parser Tests

TEST_F(ParseHLSLRootSignatureTest, InvalidParseUnexpectedEOSTest) {
  const llvm::StringLiteral Source = R"cc(
    DescriptorTable
  )cc";

  TrivialModuleLoader ModLoader;
  auto PP = CreatePP(Source, ModLoader);
  auto TokLoc = SourceLocation();

  // Test correct diagnostic produced
  Consumer->SetExpected(diag::err_hlsl_rootsig_unexpected_eos);
  hlsl::RootSignatureLexer Lexer(Source, TokLoc, *PP);

  SmallVector<hlsl::RootSignatureToken> Tokens;
  ASSERT_FALSE(Lexer.Lex(Tokens));

  SmallVector<RootElement> Elements;
  hlsl::RootSignatureParser Parser(Elements, Tokens, Diags);

  ASSERT_TRUE(Parser.Parse());

  ASSERT_TRUE(Consumer->IsSatisfied());
}

TEST_F(ParseHLSLRootSignatureTest, InvalidParseUnexpectedTokenTest) {
  const llvm::StringLiteral Source = R"cc(
    DescriptorTable()
    space
  )cc";

  TrivialModuleLoader ModLoader;
  auto PP = CreatePP(Source, ModLoader);
  auto TokLoc = SourceLocation();

  // Test correct diagnostic produced
  Consumer->SetExpected(diag::err_hlsl_rootsig_unexpected_token_kind);
  hlsl::RootSignatureLexer Lexer(Source, TokLoc, *PP);

  SmallVector<hlsl::RootSignatureToken> Tokens;
  ASSERT_FALSE(Lexer.Lex(Tokens));

  SmallVector<RootElement> Elements;
  hlsl::RootSignatureParser Parser(Elements, Tokens, Diags);

  ASSERT_TRUE(Parser.Parse());

  ASSERT_TRUE(Consumer->IsSatisfied());
}

TEST_F(ParseHLSLRootSignatureTest, InvalidParseRepeatedParamTest) {
  const llvm::StringLiteral Source = R"cc(
    DescriptorTable(
      CBV(b0, numDescriptors = 3, numDescriptors = 1)
    )
  )cc";

  TrivialModuleLoader ModLoader;
  auto PP = CreatePP(Source, ModLoader);
  auto TokLoc = SourceLocation();

  // Test correct diagnostic produced
  Consumer->SetExpected(diag::err_hlsl_rootsig_repeat_param);
  hlsl::RootSignatureLexer Lexer(Source, TokLoc, *PP);

  SmallVector<hlsl::RootSignatureToken> Tokens;
  ASSERT_FALSE(Lexer.Lex(Tokens));

  SmallVector<RootElement> Elements;
  hlsl::RootSignatureParser Parser(Elements, Tokens, Diags);

  ASSERT_TRUE(Parser.Parse());

  ASSERT_TRUE(Consumer->IsSatisfied());
}

TEST_F(ParseHLSLRootSignatureTest, InvalidParseRepeatedVisibilityTest) {
  const llvm::StringLiteral Source = R"cc(
    DescriptorTable(
      visibility = SHADER_VISIBILITY_GEOMETRY,
      visibility = SHADER_VISIBILITY_HULL
    )
  )cc";

  TrivialModuleLoader ModLoader;
  auto PP = CreatePP(Source, ModLoader);
  auto TokLoc = SourceLocation();

  // Test correct diagnostic produced
  Consumer->SetExpected(diag::err_hlsl_rootsig_repeat_param);
  hlsl::RootSignatureLexer Lexer(Source, TokLoc, *PP);

  SmallVector<hlsl::RootSignatureToken> Tokens;
  ASSERT_FALSE(Lexer.Lex(Tokens));

  SmallVector<RootElement> Elements;
  hlsl::RootSignatureParser Parser(Elements, Tokens, Diags);

  ASSERT_TRUE(Parser.Parse());

  ASSERT_TRUE(Consumer->IsSatisfied());
}

TEST_F(ParseHLSLRootSignatureTest, InvalidParseNonZeroFlagTest) {
  const llvm::StringLiteral Source = R"cc(
    DescriptorTable(
      CBV(b0, flags = 3)
    )
  )cc";

  TrivialModuleLoader ModLoader;
  auto PP = CreatePP(Source, ModLoader);
  auto TokLoc = SourceLocation();

  // Test correct diagnostic produced
  Consumer->SetExpected(diag::err_hlsl_rootsig_non_zero_flag);
  hlsl::RootSignatureLexer Lexer(Source, TokLoc, *PP);

  SmallVector<hlsl::RootSignatureToken> Tokens;
  ASSERT_FALSE(Lexer.Lex(Tokens));

  SmallVector<RootElement> Elements;
  hlsl::RootSignatureParser Parser(Elements, Tokens, Diags);

  ASSERT_TRUE(Parser.Parse());

  ASSERT_TRUE(Consumer->IsSatisfied());
}

} // anonymous namespace
