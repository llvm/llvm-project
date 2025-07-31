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
#include "clang/Basic/SourceManager.h"
#include "llvm/Support/SmallVectorMemoryBuffer.h"
#include "gtest/gtest.h"

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
  IntrusiveRefCntPtr<DiagnosticIDs> DiagID(new DiagnosticIDs);
  DiagnosticOptions DiagEngineOpts;
  DiagnosticsEngine DiagEngine(DiagID, DiagEngineOpts,
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

} // anonymous namespace
