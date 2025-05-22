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
        Consumer(new ExpectedDiagConsumer()), Diags(DiagID, DiagOpts, Consumer),
        SourceMgr(Diags, FileMgr), TargetOpts(new TargetOptions) {
    // This is an arbitrarily chosen target triple to create the target info.
    TargetOpts->Triple = "dxil";
    Target = TargetInfo::CreateTargetInfo(Diags, *TargetOpts);
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
  DiagnosticOptions DiagOpts;
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
      SRV(space = 3, offset = 32, t42, flags = 0, numDescriptors = 4),
      visibility = SHADER_VISIBILITY_PIXEL,
      Sampler(s987, space = +2, offset = DESCRIPTOR_RANGE_OFFSET_APPEND),
      UAV(u4294967294, numDescriptors = unbounded,
        flags = Descriptors_Volatile | Data_Volatile
                      | Data_Static_While_Set_At_Execute | Data_Static
                      | Descriptors_Static_Keeping_Buffer_Bounds_Checks
      )
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
  ASSERT_EQ(std::get<DescriptorTableClause>(Elem).Reg.ViewType,
            RegisterType::BReg);
  ASSERT_EQ(std::get<DescriptorTableClause>(Elem).Reg.Number, 0u);
  ASSERT_EQ(std::get<DescriptorTableClause>(Elem).NumDescriptors, 1u);
  ASSERT_EQ(std::get<DescriptorTableClause>(Elem).Space, 0u);
  ASSERT_EQ(std::get<DescriptorTableClause>(Elem).Offset,
            DescriptorTableOffsetAppend);
  ASSERT_EQ(std::get<DescriptorTableClause>(Elem).Flags,
            DescriptorRangeFlags::DataStaticWhileSetAtExecute);

  Elem = Elements[1];
  ASSERT_TRUE(std::holds_alternative<DescriptorTableClause>(Elem));
  ASSERT_EQ(std::get<DescriptorTableClause>(Elem).Type, ClauseType::SRV);
  ASSERT_EQ(std::get<DescriptorTableClause>(Elem).Reg.ViewType,
            RegisterType::TReg);
  ASSERT_EQ(std::get<DescriptorTableClause>(Elem).Reg.Number, 42u);
  ASSERT_EQ(std::get<DescriptorTableClause>(Elem).NumDescriptors, 4u);
  ASSERT_EQ(std::get<DescriptorTableClause>(Elem).Space, 3u);
  ASSERT_EQ(std::get<DescriptorTableClause>(Elem).Offset, 32u);
  ASSERT_EQ(std::get<DescriptorTableClause>(Elem).Flags,
            DescriptorRangeFlags::None);

  Elem = Elements[2];
  ASSERT_TRUE(std::holds_alternative<DescriptorTableClause>(Elem));
  ASSERT_EQ(std::get<DescriptorTableClause>(Elem).Type, ClauseType::Sampler);
  ASSERT_EQ(std::get<DescriptorTableClause>(Elem).Reg.ViewType,
            RegisterType::SReg);
  ASSERT_EQ(std::get<DescriptorTableClause>(Elem).Reg.Number, 987u);
  ASSERT_EQ(std::get<DescriptorTableClause>(Elem).NumDescriptors, 1u);
  ASSERT_EQ(std::get<DescriptorTableClause>(Elem).Space, 2u);
  ASSERT_EQ(std::get<DescriptorTableClause>(Elem).Offset,
            DescriptorTableOffsetAppend);
  ASSERT_EQ(std::get<DescriptorTableClause>(Elem).Flags,
            DescriptorRangeFlags::None);

  Elem = Elements[3];
  ASSERT_TRUE(std::holds_alternative<DescriptorTableClause>(Elem));
  ASSERT_EQ(std::get<DescriptorTableClause>(Elem).Type, ClauseType::UAV);
  ASSERT_EQ(std::get<DescriptorTableClause>(Elem).Reg.ViewType,
            RegisterType::UReg);
  ASSERT_EQ(std::get<DescriptorTableClause>(Elem).Reg.Number, 4294967294u);
  ASSERT_EQ(std::get<DescriptorTableClause>(Elem).NumDescriptors,
            NumDescriptorsUnbounded);
  ASSERT_EQ(std::get<DescriptorTableClause>(Elem).Space, 0u);
  ASSERT_EQ(std::get<DescriptorTableClause>(Elem).Offset,
            DescriptorTableOffsetAppend);
  ASSERT_EQ(std::get<DescriptorTableClause>(Elem).Flags,
            DescriptorRangeFlags::ValidFlags);

  Elem = Elements[4];
  ASSERT_TRUE(std::holds_alternative<DescriptorTable>(Elem));
  ASSERT_EQ(std::get<DescriptorTable>(Elem).NumClauses, (uint32_t)4);
  ASSERT_EQ(std::get<DescriptorTable>(Elem).Visibility,
            ShaderVisibility::Pixel);

  // Empty Descriptor Table
  Elem = Elements[5];
  ASSERT_TRUE(std::holds_alternative<DescriptorTable>(Elem));
  ASSERT_EQ(std::get<DescriptorTable>(Elem).NumClauses, 0u);
  ASSERT_EQ(std::get<DescriptorTable>(Elem).Visibility, ShaderVisibility::All);

  ASSERT_TRUE(Consumer->isSatisfied());
}

TEST_F(ParseHLSLRootSignatureTest, ValidSamplerFlagsTest) {
  // This test will checks we can set the valid enum for Sampler descriptor
  // range flags
  const llvm::StringLiteral Source = R"cc(
    DescriptorTable(Sampler(s0, flags = DESCRIPTORS_VOLATILE))
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

  RootElement Elem = Elements[0];
  ASSERT_TRUE(std::holds_alternative<DescriptorTableClause>(Elem));
  ASSERT_EQ(std::get<DescriptorTableClause>(Elem).Type, ClauseType::Sampler);
  ASSERT_EQ(std::get<DescriptorTableClause>(Elem).Flags,
            DescriptorRangeFlags::ValidSamplerFlags);

  ASSERT_TRUE(Consumer->isSatisfied());
}

TEST_F(ParseHLSLRootSignatureTest, ValidParseRootConsantsTest) {
  const llvm::StringLiteral Source = R"cc(
    RootConstants(num32BitConstants = 1, b0),
    RootConstants(b42, space = 3, num32BitConstants = 4294967295,
      visibility = SHADER_VISIBILITY_HULL
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

  ASSERT_EQ(Elements.size(), 2u);

  RootElement Elem = Elements[0];
  ASSERT_TRUE(std::holds_alternative<RootConstants>(Elem));
  ASSERT_EQ(std::get<RootConstants>(Elem).Num32BitConstants, 1u);
  ASSERT_EQ(std::get<RootConstants>(Elem).Reg.ViewType, RegisterType::BReg);
  ASSERT_EQ(std::get<RootConstants>(Elem).Reg.Number, 0u);
  ASSERT_EQ(std::get<RootConstants>(Elem).Space, 0u);
  ASSERT_EQ(std::get<RootConstants>(Elem).Visibility, ShaderVisibility::All);

  Elem = Elements[1];
  ASSERT_TRUE(std::holds_alternative<RootConstants>(Elem));
  ASSERT_EQ(std::get<RootConstants>(Elem).Num32BitConstants, 4294967295u);
  ASSERT_EQ(std::get<RootConstants>(Elem).Reg.ViewType, RegisterType::BReg);
  ASSERT_EQ(std::get<RootConstants>(Elem).Reg.Number, 42u);
  ASSERT_EQ(std::get<RootConstants>(Elem).Space, 3u);
  ASSERT_EQ(std::get<RootConstants>(Elem).Visibility, ShaderVisibility::Hull);

  ASSERT_TRUE(Consumer->isSatisfied());
}

TEST_F(ParseHLSLRootSignatureTest, ValidParseRootFlagsTest) {
  const llvm::StringLiteral Source = R"cc(
    RootFlags(),
    RootFlags(0),
    RootFlags(
      deny_domain_shader_root_access |
      deny_pixel_shader_root_access |
      local_root_signature |
      cbv_srv_uav_heap_directly_indexed |
      deny_amplification_shader_root_access |
      deny_geometry_shader_root_access |
      deny_hull_shader_root_access |
      deny_mesh_shader_root_access |
      allow_stream_output |
      sampler_heap_directly_indexed |
      allow_input_assembler_input_layout |
      deny_vertex_shader_root_access
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

  ASSERT_EQ(Elements.size(), 3u);

  RootElement Elem = Elements[0];
  ASSERT_TRUE(std::holds_alternative<RootFlags>(Elem));
  ASSERT_EQ(std::get<RootFlags>(Elem), RootFlags::None);

  Elem = Elements[1];
  ASSERT_TRUE(std::holds_alternative<RootFlags>(Elem));
  ASSERT_EQ(std::get<RootFlags>(Elem), RootFlags::None);

  Elem = Elements[2];
  ASSERT_TRUE(std::holds_alternative<RootFlags>(Elem));
  ASSERT_EQ(std::get<RootFlags>(Elem), RootFlags::ValidFlags);

  ASSERT_TRUE(Consumer->isSatisfied());
}

TEST_F(ParseHLSLRootSignatureTest, ValidParseRootDescriptorsTest) {
  const llvm::StringLiteral Source = R"cc(
    CBV(b0),
    SRV(t42),
    UAV(u34893247)
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

  ASSERT_EQ(Elements.size(), 3u);

  RootElement Elem = Elements[0];
  ASSERT_TRUE(std::holds_alternative<RootDescriptor>(Elem));
  ASSERT_EQ(std::get<RootDescriptor>(Elem).Type, DescriptorType::CBuffer);
  ASSERT_EQ(std::get<RootDescriptor>(Elem).Reg.ViewType, RegisterType::BReg);
  ASSERT_EQ(std::get<RootDescriptor>(Elem).Reg.Number, 0u);

  Elem = Elements[1];
  ASSERT_TRUE(std::holds_alternative<RootDescriptor>(Elem));
  ASSERT_EQ(std::get<RootDescriptor>(Elem).Type, DescriptorType::SRV);
  ASSERT_EQ(std::get<RootDescriptor>(Elem).Reg.ViewType, RegisterType::TReg);
  ASSERT_EQ(std::get<RootDescriptor>(Elem).Reg.Number, 42u);

  Elem = Elements[2];
  ASSERT_TRUE(std::holds_alternative<RootDescriptor>(Elem));
  ASSERT_EQ(std::get<RootDescriptor>(Elem).Type, DescriptorType::UAV);
  ASSERT_EQ(std::get<RootDescriptor>(Elem).Reg.ViewType, RegisterType::UReg);
  ASSERT_EQ(std::get<RootDescriptor>(Elem).Reg.Number, 34893247u);

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

TEST_F(ParseHLSLRootSignatureTest, InvalidMissingDTParameterTest) {
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

TEST_F(ParseHLSLRootSignatureTest, InvalidMissingRDParameterTest) {
  // This test will check that the parsing fails due a mandatory
  // parameter (register) not being specified
  const llvm::StringLiteral Source = R"cc(
    SRV()
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

TEST_F(ParseHLSLRootSignatureTest, InvalidMissingRCParameterTest) {
  // This test will check that the parsing fails due a mandatory
  // parameter (num32BitConstants) not being specified
  const llvm::StringLiteral Source = R"cc(
    RootConstants(b0)
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

TEST_F(ParseHLSLRootSignatureTest, InvalidRepeatedMandatoryDTParameterTest) {
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

TEST_F(ParseHLSLRootSignatureTest, InvalidRepeatedMandatoryRCParameterTest) {
  // This test will check that the parsing fails due the same mandatory
  // parameter being specified multiple times
  const llvm::StringLiteral Source = R"cc(
    RootConstants(num32BitConstants = 32, num32BitConstants = 24)
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

TEST_F(ParseHLSLRootSignatureTest, InvalidRepeatedOptionalDTParameterTest) {
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

TEST_F(ParseHLSLRootSignatureTest, InvalidRepeatedOptionalRCParameterTest) {
  // This test will check that the parsing fails due the same optional
  // parameter being specified multiple times
  const llvm::StringLiteral Source = R"cc(
    RootConstants(
      visibility = Shader_Visibility_All,
      b0, num32BitConstants = 1,
      visibility = Shader_Visibility_Pixel
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

TEST_F(ParseHLSLRootSignatureTest, InvalidNonZeroFlagsTest) {
  // This test will check that parsing fails when a non-zero integer literal
  // is given to flags
  const llvm::StringLiteral Source = R"cc(
    DescriptorTable(
      CBV(b0, flags = 3)
    )
  )cc";

  TrivialModuleLoader ModLoader;
  auto PP = createPP(Source, ModLoader);
  auto TokLoc = SourceLocation();

  hlsl::RootSignatureLexer Lexer(Source, TokLoc);
  SmallVector<RootElement> Elements;
  hlsl::RootSignatureParser Parser(Elements, Lexer, *PP);

  // Test correct diagnostic produced
  Consumer->setExpected(diag::err_hlsl_rootsig_non_zero_flag);
  ASSERT_TRUE(Parser.parse());

  ASSERT_TRUE(Consumer->isSatisfied());
}

} // anonymous namespace
