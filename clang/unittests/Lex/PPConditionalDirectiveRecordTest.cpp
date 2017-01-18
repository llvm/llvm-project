//===- unittests/Lex/PPConditionalDirectiveRecordTest.cpp-PP directive tests =//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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
      DiagID(new DiagnosticIDs()),
      Diags(DiagID, new DiagnosticOptions, new IgnoringDiagConsumer()),
      SourceMgr(Diags, FileMgr),
      TargetOpts(new TargetOptions)
  {
    TargetOpts->Triple = "x86_64-apple-darwin11.1.0";
    Target = TargetInfo::CreateTargetInfo(Diags, TargetOpts);
  }

  FileSystemOptions FileMgrOpts;
  FileManager FileMgr;
  IntrusiveRefCntPtr<DiagnosticIDs> DiagID;
  DiagnosticsEngine Diags;
  SourceManager SourceMgr;
  LangOptions LangOpts;
  std::shared_ptr<TargetOptions> TargetOpts;
  IntrusiveRefCntPtr<TargetInfo> Target;
};

class VoidModuleLoader : public ModuleLoader {
  ModuleLoadResult loadModule(SourceLocation ImportLoc, 
                              ModuleIdPath Path,
                              Module::NameVisibilityKind Visibility,
                              bool IsInclusionDirective) override {
    return ModuleLoadResult();
  }

  void makeModuleVisible(Module *Mod,
                         Module::NameVisibilityKind Visibility,
                         SourceLocation ImportLoc) override { }

  GlobalModuleIndex *loadGlobalModuleIndex(SourceLocation TriggerLoc) override
    { return nullptr; }
  bool lookupMissingImports(StringRef Name, SourceLocation TriggerLoc) override
    { return 0; }
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

  VoidModuleLoader ModLoader;
  HeaderSearch HeaderInfo(std::make_shared<HeaderSearchOptions>(), SourceMgr,
                          Diags, LangOpts, Target.get());
  Preprocessor PP(std::make_shared<PreprocessorOptions>(), Diags, LangOpts,
                  SourceMgr, HeaderInfo, ModLoader,
                  /*IILookup =*/nullptr,
                  /*OwnsHeaderSearch =*/false);
  PP.Initialize(*Target);
  PPConditionalDirectiveRecord *
    PPRec = new PPConditionalDirectiveRecord(SourceMgr);
  PP.addPPCallbacks(std::unique_ptr<PPCallbacks>(PPRec));
  PP.EnterMainSourceFile();

  std::vector<Token> toks;
  while (1) {
    Token tok;
    PP.Lex(tok);
    if (tok.is(tok::eof))
      break;
    toks.push_back(tok);
  }

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
