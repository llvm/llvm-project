//===- unittests/Lex/PPMemoryAllocationsTest.cpp - ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===--------------------------------------------------------------===//

#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/DiagnosticOptions.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/LangOptions.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Basic/TargetOptions.h"
#include "clang/Lex/HeaderSearch.h"
#include "clang/Lex/HeaderSearchOptions.h"
#include "clang/Lex/ModuleLoader.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Lex/PreprocessorOptions.h"
#include "gtest/gtest.h"

using namespace clang;

namespace {

class PPMemoryAllocationsTest : public ::testing::Test {
protected:
  PPMemoryAllocationsTest()
      : FileMgr(FileMgrOpts),
        Diags(DiagnosticIDs::create(), DiagOpts, new IgnoringDiagConsumer()),
        SourceMgr(Diags, FileMgr), TargetOpts(new TargetOptions) {
    TargetOpts->Triple = "x86_64-apple-darwin11.1.0";
    Target = TargetInfo::CreateTargetInfo(Diags, *TargetOpts);
  }

  FileSystemOptions FileMgrOpts;
  FileManager FileMgr;
  DiagnosticOptions DiagOpts;
  DiagnosticsEngine Diags;
  SourceManager SourceMgr;
  LangOptions LangOpts;
  std::shared_ptr<TargetOptions> TargetOpts;
  IntrusiveRefCntPtr<TargetInfo> Target;
};

TEST_F(PPMemoryAllocationsTest, PPMacroDefinesAllocations) {
  std::string Source;
  size_t NumMacros = 1000000;
  {
    llvm::raw_string_ostream SourceOS(Source);

    // Create a combination of 1 or 3 token macros.
    for (size_t I = 0; I < NumMacros; ++I) {
      SourceOS << "#define MACRO_ID_" << I << " ";
      if ((I % 2) == 0)
        SourceOS << "(" << I << ")";
      else
        SourceOS << I;
      SourceOS << "\n";
    }
  }

  std::unique_ptr<llvm::MemoryBuffer> Buf =
      llvm::MemoryBuffer::getMemBuffer(Source);
  SourceMgr.setMainFileID(SourceMgr.createFileID(std::move(Buf)));

  HeaderSearchOptions HSOpts;
  TrivialModuleLoader ModLoader;
  HeaderSearch HeaderInfo(HSOpts, SourceMgr, Diags, LangOpts, Target.get());
  PreprocessorOptions PPOpts;
  Preprocessor PP(PPOpts, Diags, LangOpts, SourceMgr, HeaderInfo, ModLoader,
                  /*IILookup =*/nullptr,
                  /*OwnsHeaderSearch =*/false);
  PP.Initialize(*Target);
  PP.EnterMainSourceFile();

  PP.LexTokensUntilEOF();

  size_t NumAllocated = PP.getPreprocessorAllocator().getBytesAllocated();
  float BytesPerDefine = float(NumAllocated) / float(NumMacros);
  llvm::errs() << "Num preprocessor allocations for " << NumMacros
               << " #define: " << NumAllocated << "\n";
  llvm::errs() << "Bytes per #define: " << BytesPerDefine << "\n";
  // On arm64-apple-macos, we get around 120 bytes per define.
  // Assume a reasonable upper bound based on that number that we don't want
  // to exceed when storing information about a macro #define with 1 or 3
  // tokens.
  EXPECT_LT(BytesPerDefine, 130.0f);
}

} // anonymous namespace
