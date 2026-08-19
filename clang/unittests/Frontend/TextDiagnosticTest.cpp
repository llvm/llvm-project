//===- unittests/Frontend/TextDiagnosticTest.cpp - ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Frontend/TextDiagnostic.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/LangOptions.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Frontend/DiagnosticRenderer.h"
#include "clang/Frontend/TextDiagnosticPrinter.h"
#include "llvm/Support/SmallVectorMemoryBuffer.h"
#include "gtest/gtest.h"
#include <optional>

using namespace llvm;
using namespace clang;

namespace {

/// Prints a diagnostic with the given DiagnosticOptions and the given
/// SourceLocation and returns the printed diagnostic text.
static std::string PrintDiag(DiagnosticOptions &Opts, FullSourceLoc Loc) {
  std::string Out;
  llvm::raw_string_ostream OS(Out);
  clang::LangOptions LangOpts;
  // Owned by TextDiagnostic.
  TextDiagnostic Diag(OS, LangOpts, Opts);
  // Emit a dummy diagnostic that is just 'message'.
  Diag.emitDiagnostic(Loc, DiagnosticsEngine::Level::Warning, "message",
                      /*Ranges=*/{}, /*FixItHints=*/{});
  return Out;
}

TEST(TextDiagnostic, ShowLine) {
  // Create dummy FileManager and SourceManager.
  FileSystemOptions FSOpts;
  FileManager FileMgr(FSOpts);
  DiagnosticOptions DiagEngineOpts;
  DiagnosticsEngine DiagEngine(DiagnosticIDs::create(), DiagEngineOpts,
                               new IgnoringDiagConsumer());
  SourceManager SrcMgr(DiagEngine, FileMgr);

  // Create a dummy file with some contents to produce a test SourceLocation.
  const llvm::StringRef file_path = "main.cpp";
  const llvm::StringRef main_file_contents = "some\nsource\ncode\n";
  const clang::FileEntryRef fe = FileMgr.getVirtualFileRef(
      file_path,
      /*Size=*/static_cast<off_t>(main_file_contents.size()),
      /*ModificationTime=*/0);

  llvm::SmallVector<char, 64> buffer;
  buffer.append(main_file_contents.begin(), main_file_contents.end());
  auto file_contents = std::make_unique<llvm::SmallVectorMemoryBuffer>(
      std::move(buffer), file_path, /*RequiresNullTerminator=*/false);
  SrcMgr.overrideFileContents(fe, std::move(file_contents));

  // Create the actual file id and use it as the main file.
  clang::FileID fid =
      SrcMgr.createFileID(fe, SourceLocation(), clang::SrcMgr::C_User);
  SrcMgr.setMainFileID(fid);

  // Create the source location for the test diagnostic.
  FullSourceLoc Loc(SrcMgr.translateLineCol(fid, /*Line=*/1, /*Col=*/2),
                    SrcMgr);

  DiagnosticOptions DiagOpts;
  DiagOpts.ShowLine = true;
  DiagOpts.ShowColumn = true;
  // Hide printing the source line/caret to make the diagnostic shorter and it's
  // not relevant for this test.
  DiagOpts.ShowCarets = false;
  EXPECT_EQ("main.cpp:1:2: warning: message\n", PrintDiag(DiagOpts, Loc));

  // Check that ShowLine doesn't influence the Vi/MSVC diagnostic formats as its
  // a Clang-specific diagnostic option.
  DiagOpts.setFormat(TextDiagnosticFormat::Vi);
  DiagOpts.ShowLine = false;
  EXPECT_EQ("main.cpp +1:2: warning: message\n", PrintDiag(DiagOpts, Loc));

  DiagOpts.setFormat(TextDiagnosticFormat::MSVC);
  DiagOpts.ShowLine = false;
  EXPECT_EQ("main.cpp(1,2): warning: message\n", PrintDiag(DiagOpts, Loc));

  // Reset back to the Clang format.
  DiagOpts.setFormat(TextDiagnosticFormat::Clang);

  // Hide line number but show column.
  DiagOpts.ShowLine = false;
  EXPECT_EQ("main.cpp:2: warning: message\n", PrintDiag(DiagOpts, Loc));

  // Show line number but hide column.
  DiagOpts.ShowLine = true;
  DiagOpts.ShowColumn = false;
  EXPECT_EQ("main.cpp:1: warning: message\n", PrintDiag(DiagOpts, Loc));
}

struct ShowLevelNoLocationTest
    : public ::testing::TestWithParam<bool /* ShowLevel */> {};

TEST_P(ShowLevelNoLocationTest, LevelPrefixRespected) {
  bool ShowLevel = GetParam();
  DiagnosticOptions DiagOpts;
  DiagOpts.ShowLevel = ShowLevel;
  std::string Output;
  llvm::raw_string_ostream OS(Output);
  TextDiagnosticPrinter Printer(OS, DiagOpts);
  DiagnosticsEngine Diags(DiagnosticIDs::create(), DiagOpts, &Printer,
                          /*ShouldOwnClient=*/false);
  // Report without a SourceLocation, exercises the no-location path in
  // TextDiagnosticPrinter::HandleDiagnostic.
  unsigned ID = Diags.getCustomDiagID(DiagnosticsEngine::Error, "%0");
  Diags.Report(ID) << "message";
  if (ShowLevel)
    EXPECT_EQ(Output, "error: message\n");
  else
    EXPECT_EQ(Output, "message\n");
}

INSTANTIATE_TEST_SUITE_P(ShowLevelNoLocation, ShowLevelNoLocationTest,
                         ::testing::Bool());

// Creates a virtual file with the given contents and returns its FileID.
static FileID makeFile(FileManager &FileMgr, SourceManager &SrcMgr,
                       StringRef Path, StringRef Contents) {
  FileEntryRef FE = FileMgr.getVirtualFileRef(
      Path, /*Size=*/static_cast<off_t>(Contents.size()),
      /*ModificationTime=*/0);
  SmallVector<char, 64> Buffer(Contents.begin(), Contents.end());
  SrcMgr.overrideFileContents(FE, std::make_unique<SmallVectorMemoryBuffer>(
                                      std::move(Buffer), Path,
                                      /*RequiresNullTerminator=*/false));
  return SrcMgr.createFileID(FE, SourceLocation(), SrcMgr::C_User);
}

TEST(DiagnosticRenderer, GetExpansionRangeInFileTest) {
  FileSystemOptions FSOpts;
  FileManager FileMgr(FSOpts);
  DiagnosticOptions DiagEngineOpts;
  DiagnosticsEngine DiagEngine(DiagnosticIDs::create(), DiagEngineOpts,
                               new IgnoringDiagConsumer());
  SourceManager SM(DiagEngine, FileMgr);

  FileID FID = makeFile(FileMgr, SM, "main.cpp", "some\nsource\ncode\n");
  FileID OtherFID = makeFile(FileMgr, SM, "other.cpp", "other\n");
  SM.setMainFileID(FID);

  auto Loc = [&](unsigned Line, unsigned Col) {
    return SM.translateLineCol(FID, Line, Col);
  };

  const SourceLocation L1C1 = Loc(/*Line=*/1, /*Col=*/1);
  const SourceLocation L1C3 = Loc(/*Line=*/1, /*Col=*/3);

  // An invalid range is rejected.
  EXPECT_FALSE(getExpansionRangeInFile(CharSourceRange(), FID, SM));

  // A char range stays a char range.
  std::optional<CharSourceRange> CharR = getExpansionRangeInFile(
      CharSourceRange::getCharRange(L1C1, L1C3), FID, SM);
  ASSERT_TRUE(CharR);
  EXPECT_TRUE(CharR->isCharRange());

  // A token range stays a token range.
  std::optional<CharSourceRange> TokR = getExpansionRangeInFile(
      CharSourceRange::getTokenRange(L1C1, L1C3), FID, SM);
  ASSERT_TRUE(TokR);
  EXPECT_TRUE(TokR->isTokenRange());

  // A reversed range (begin lies after end) is rejected.
  EXPECT_FALSE(getExpansionRangeInFile(
      CharSourceRange::getCharRange(L1C3, L1C1), FID, SM));

  // The endpoints are compared as-is, so a reversed token range is rejected
  // too, even though extending its end token would order the offsets.
  EXPECT_FALSE(getExpansionRangeInFile(
      CharSourceRange::getTokenRange(L1C3, L1C1), FID, SM));

  // A range with an endpoint in another file is rejected.
  SourceLocation OtherLoc = SM.getLocForStartOfFile(OtherFID);
  EXPECT_FALSE(getExpansionRangeInFile(
      CharSourceRange::getTokenRange(L1C1, OtherLoc), FID, SM));

  {
    const SourceLocation L2C1 = Loc(/*Line=*/2, /*Col=*/1);
    const SourceLocation L2C6 = Loc(/*Line=*/2, /*Col=*/6);

    // Pretend that "source" expands "some".
    SourceLocation MacroLoc = SM.createExpansionLoc(
        /*SpellingLoc=*/L1C1, /*ExpansionLocStart=*/L2C1,
        /*ExpansionLocEnd=*/L2C6, /*Length=*/4);
    ASSERT_TRUE(MacroLoc.isMacroID());
    ASSERT_EQ(SM.getSpellingLoc(MacroLoc), L1C1);
    ASSERT_EQ(SM.getExpansionLoc(MacroLoc), L2C1);

    // A macro-expanded range is remapped to its expansion in the file.
    // A location inside the macro maps back to that file range.
    auto MacroToken = CharSourceRange::getTokenRange(MacroLoc, MacroLoc);
    auto MacroR = getExpansionRangeInFile(MacroToken, FID, SM);
    ASSERT_TRUE(MacroR);
    EXPECT_EQ(SM.getFileID(MacroR->getBegin()), FID);
    EXPECT_EQ(SM.getFileID(MacroR->getEnd()), FID);

    // The range is a file range.
    EXPECT_TRUE(MacroR->getBegin().isFileID());
    EXPECT_TRUE(MacroR->getEnd().isFileID());

    // The range is the expansion range.
    EXPECT_EQ(MacroR->getBegin(), L2C1);
    EXPECT_EQ(MacroR->getEnd(), L2C6);
  }
}

} // anonymous namespace
