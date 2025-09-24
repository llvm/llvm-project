//===- unittests/Lex/PPConditionalDirectiveRecordTest.cpp-PP directive tests =//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Lex/PPConditionalDirectiveRecord.h"
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

// The test fixture.
class PPConditionalDirectiveRecordTest : public ::testing::Test {
protected:
  PPConditionalDirectiveRecordTest()
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

TEST_F(PPConditionalDirectiveRecordTest, PPRecAPI) {
  const char *source =
      "0 1\n"
      "#if 1\n"
      "2\n"
      "#ifndef BB\n"
      "3 4\n"
      "#else\n"
      "#endif\n"
      "5\n"
      "#endif\n"
      "6\n"
      "#if 1\n"
      "7\n"
      "#if 1\n"
      "#endif\n"
      "8\n"
      "#endif\n"
      "9\n";

  std::unique_ptr<llvm::MemoryBuffer> Buf =
      llvm::MemoryBuffer::getMemBuffer(source);
  SourceMgr.setMainFileID(SourceMgr.createFileID(std::move(Buf)));

  HeaderSearchOptions HSOpts;
  TrivialModuleLoader ModLoader;
  HeaderSearch HeaderInfo(HSOpts, SourceMgr, Diags, LangOpts, Target.get());
  PreprocessorOptions PPOpts;
  Preprocessor PP(PPOpts, Diags, LangOpts, SourceMgr, HeaderInfo, ModLoader,
                  /*IILookup =*/nullptr,
                  /*OwnsHeaderSearch =*/false);
  PP.Initialize(*Target);
  PPConditionalDirectiveRecord *
    PPRec = new PPConditionalDirectiveRecord(SourceMgr);
  PP.addPPCallbacks(std::unique_ptr<PPCallbacks>(PPRec));
  PP.EnterMainSourceFile();

  std::vector<Token> toks;
  PP.LexTokensUntilEOF(&toks);

  // Make sure we got the tokens that we expected.
  ASSERT_EQ(10U, toks.size());
  
  EXPECT_FALSE(PPRec->rangeIntersectsConditionalDirective(
                    SourceRange(toks[0].getLocation(), toks[1].getLocation())));
  EXPECT_TRUE(PPRec->rangeIntersectsConditionalDirective(
                    SourceRange(toks[0].getLocation(), toks[2].getLocation())));
  EXPECT_FALSE(PPRec->rangeIntersectsConditionalDirective(
                    SourceRange(toks[3].getLocation(), toks[4].getLocation())));
  EXPECT_TRUE(PPRec->rangeIntersectsConditionalDirective(
                    SourceRange(toks[1].getLocation(), toks[5].getLocation())));
  EXPECT_TRUE(PPRec->rangeIntersectsConditionalDirective(
                    SourceRange(toks[2].getLocation(), toks[6].getLocation())));
  EXPECT_FALSE(PPRec->rangeIntersectsConditionalDirective(
                    SourceRange(toks[2].getLocation(), toks[5].getLocation())));
  EXPECT_FALSE(PPRec->rangeIntersectsConditionalDirective(
                    SourceRange(toks[0].getLocation(), toks[6].getLocation())));
  EXPECT_TRUE(PPRec->rangeIntersectsConditionalDirective(
                    SourceRange(toks[2].getLocation(), toks[8].getLocation())));
  EXPECT_FALSE(PPRec->rangeIntersectsConditionalDirective(
                    SourceRange(toks[0].getLocation(), toks[9].getLocation())));

  EXPECT_TRUE(PPRec->areInDifferentConditionalDirectiveRegion(
                    toks[0].getLocation(), toks[2].getLocation()));
  EXPECT_FALSE(PPRec->areInDifferentConditionalDirectiveRegion(
                    toks[3].getLocation(), toks[4].getLocation()));
  EXPECT_TRUE(PPRec->areInDifferentConditionalDirectiveRegion(
                    toks[1].getLocation(), toks[5].getLocation()));
  EXPECT_TRUE(PPRec->areInDifferentConditionalDirectiveRegion(
                    toks[2].getLocation(), toks[0].getLocation()));
  EXPECT_FALSE(PPRec->areInDifferentConditionalDirectiveRegion(
                    toks[4].getLocation(), toks[3].getLocation()));
  EXPECT_TRUE(PPRec->areInDifferentConditionalDirectiveRegion(
                    toks[5].getLocation(), toks[1].getLocation()));
}

} // anonymous namespace
