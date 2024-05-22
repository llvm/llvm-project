//===-- ReplayPreambleTests.cpp -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// These tests cover clangd's logic to replay PP events from preamble to
// clang-tidy checks.
//
//===----------------------------------------------------------------------===//

#include "../../clang-tidy/ClangTidyCheck.h"
#include "../../clang-tidy/ClangTidyModule.h"
#include "../../clang-tidy/ClangTidyModuleRegistry.h"
#include "AST.h"
#include "Config.h"
#include "Diagnostics.h"
#include "ParsedAST.h"
#include "SourceCode.h"
#include "TestTU.h"
#include "TidyProvider.h"
#include "support/Context.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/Basic/FileEntry.h"
#include "clang/Basic/LLVM.h"
#include "clang/Basic/Module.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/TokenKinds.h"
#include "clang/Lex/PPCallbacks.h"
#include "clang/Lex/Token.h"
#include "clang/Tooling/Syntax/Tokens.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Registry.h"
#include "llvm/Testing/Annotations/Annotations.h"
#include "gmock/gmock-matchers.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <cstddef>
#include <memory>
#include <vector>

namespace clang::clangd {
namespace {
struct Inclusion {
  Inclusion(const SourceManager &SM, SourceLocation HashLoc,
            const Token &IncludeTok, llvm::StringRef FileName, bool IsAngled,
            CharSourceRange FilenameRange)
      : HashOffset(SM.getDecomposedLoc(HashLoc).second), IncTok(IncludeTok),
        IncDirective(IncludeTok.getIdentifierInfo()->getName()),
        FileNameOffset(SM.getDecomposedLoc(FilenameRange.getBegin()).second),
        FileName(FileName), IsAngled(IsAngled) {
    EXPECT_EQ(
        toSourceCode(SM, FilenameRange.getAsRange()).drop_back().drop_front(),
        FileName);
  }
  size_t HashOffset;
  syntax::Token IncTok;
  llvm::StringRef IncDirective;
  size_t FileNameOffset;
  llvm::StringRef FileName;
  bool IsAngled;
};
static std::vector<Inclusion> Includes;
static std::vector<syntax::Token> SkippedFiles;
struct ReplayPreamblePPCallback : public PPCallbacks {
  const SourceManager &SM;
  explicit ReplayPreamblePPCallback(const SourceManager &SM) : SM(SM) {}

  void InclusionDirective(SourceLocation HashLoc, const Token &IncludeTok,
                          StringRef FileName, bool IsAngled,
                          CharSourceRange FilenameRange, OptionalFileEntryRef,
                          StringRef, StringRef, const clang::Module *, bool,
                          SrcMgr::CharacteristicKind) override {
    Includes.emplace_back(SM, HashLoc, IncludeTok, FileName, IsAngled,
                          FilenameRange);
  }

  void FileSkipped(const FileEntryRef &, const Token &FilenameTok,
                   SrcMgr::CharacteristicKind) override {
    SkippedFiles.emplace_back(FilenameTok);
  }
};
struct ReplayPreambleCheck : public tidy::ClangTidyCheck {
  ReplayPreambleCheck(StringRef Name, tidy::ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context) {}
  void registerPPCallbacks(const SourceManager &SM, Preprocessor *PP,
                           Preprocessor *ModuleExpanderPP) override {
    PP->addPPCallbacks(::std::make_unique<ReplayPreamblePPCallback>(SM));
  }
};
llvm::StringLiteral CheckName = "replay-preamble-check";
struct ReplayPreambleModule : public tidy::ClangTidyModule {
  void
  addCheckFactories(tidy::ClangTidyCheckFactories &CheckFactories) override {
    CheckFactories.registerCheck<ReplayPreambleCheck>(CheckName);
  }
};
static tidy::ClangTidyModuleRegistry::Add<ReplayPreambleModule>
    X("replay-preamble-module", "");

MATCHER_P(rangeIs, R, "") {
  return arg.beginOffset() == R.Begin && arg.endOffset() == R.End;
}

TEST(ReplayPreambleTest, IncludesAndSkippedFiles) {
  TestTU TU;
  // This check records inclusion directives replayed by clangd.
  TU.ClangTidyProvider = addTidyChecks(CheckName);
  llvm::Annotations Test(R"cpp(
    $hash^#$include[[import]] $filebegin^"$filerange[[bar.h]]"
    $hash^#$include[[include_next]] $filebegin^"$filerange[[baz.h]]"
    $hash^#$include[[include]] $filebegin^<$filerange[[a.h]]>)cpp");
  llvm::StringRef Code = Test.code();
  TU.Code = Code.str();
  TU.AdditionalFiles["bar.h"] = "";
  TU.AdditionalFiles["baz.h"] = "";
  TU.AdditionalFiles["a.h"] = "";
  // Since we are also testing #import directives, and they don't make much
  // sense in c++ (also they actually break on windows), just set language to
  // obj-c.
  TU.ExtraArgs = {"-isystem.", "-xobjective-c"};

  // Allow the check to run even though not marked as fast.
  Config Cfg;
  Cfg.Diagnostics.ClangTidy.FastCheckFilter = Config::FastCheckPolicy::Loose;
  WithContextValue WithCfg(Config::Key, std::move(Cfg));

  const auto &AST = TU.build();
  const auto &SM = AST.getSourceManager();

  auto HashLocs = Test.points("hash");
  ASSERT_EQ(HashLocs.size(), Includes.size());
  auto IncludeRanges = Test.ranges("include");
  ASSERT_EQ(IncludeRanges.size(), Includes.size());
  auto FileBeginLocs = Test.points("filebegin");
  ASSERT_EQ(FileBeginLocs.size(), Includes.size());
  auto FileRanges = Test.ranges("filerange");
  ASSERT_EQ(FileRanges.size(), Includes.size());

  ASSERT_EQ(SkippedFiles.size(), Includes.size());
  for (size_t I = 0; I < Includes.size(); ++I) {
    const auto &Inc = Includes[I];

    EXPECT_EQ(Inc.HashOffset, HashLocs[I]);

    auto IncRange = IncludeRanges[I];
    EXPECT_THAT(Inc.IncTok.range(SM), rangeIs(IncRange));
    EXPECT_EQ(Inc.IncTok.kind(), tok::identifier);
    EXPECT_EQ(Inc.IncDirective,
              Code.substr(IncRange.Begin, IncRange.End - IncRange.Begin));

    EXPECT_EQ(Inc.FileNameOffset, FileBeginLocs[I]);
    EXPECT_EQ(Inc.IsAngled, Code[FileBeginLocs[I]] == '<');

    auto FileRange = FileRanges[I];
    EXPECT_EQ(Inc.FileName,
              Code.substr(FileRange.Begin, FileRange.End - FileRange.Begin));

    EXPECT_EQ(SM.getDecomposedLoc(SkippedFiles[I].location()).second,
              Inc.FileNameOffset);
    // This also contains quotes/angles so increment the range by one from both
    // sides.
    EXPECT_EQ(
        SkippedFiles[I].text(SM),
        Code.substr(FileRange.Begin - 1, FileRange.End - FileRange.Begin + 2));
    EXPECT_EQ(SkippedFiles[I].kind(), tok::header_name);
  }
}
} // namespace
} // namespace clang::clangd
