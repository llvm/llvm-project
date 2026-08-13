//===- SerializedDiagnosticsLargeFixitTest.cpp ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Regression test for the RECORD_FIXIT text-size field overflow. Historically
// the abbreviation used Fixed,16 for the FixIt's CodeToInsert size, which
// silently caps payloads at 65535 bytes and asserts in BitstreamWriter::Emit
// ("High bits set!") on +assertions builds. RECORD_DIAG already uses VBR,16
// for the same purpose; this test pins the equivalent behavior for
// RECORD_FIXIT so the two stay in sync.
//
//===----------------------------------------------------------------------===//

#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/DiagnosticOptions.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/LangOptions.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Frontend/SerializedDiagnosticPrinter.h"
#include "clang/Frontend/SerializedDiagnosticReader.h"
#include "clang/Frontend/SerializedDiagnostics.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SmallVectorMemoryBuffer.h"
#include "gtest/gtest.h"

using namespace clang;
using namespace llvm;

namespace {

// Reader that captures the text of every FixIt it sees.
class CapturingReader : public serialized_diags::SerializedDiagnosticReader {
public:
  std::vector<std::string> FixItTexts;

protected:
  std::error_code visitFixitRecord(const serialized_diags::Location &Start,
                                   const serialized_diags::Location &End,
                                   StringRef Text) override {
    FixItTexts.emplace_back(Text);
    return {};
  }
};

TEST(SerializedDiagnostics, LargeFixItRoundTrips) {
  // Build a temp output path for the .dia file.
  SmallString<128> TmpPath;
  ASSERT_FALSE(
      sys::fs::createTemporaryFile("sdiag-large-fixit", "dia", TmpPath));
  auto Cleanup = llvm::scope_exit([&] { sys::fs::remove(TmpPath); });

  // Build a 70 KiB FixIt payload — comfortably above the historical 65535
  // limit but small enough to keep the test cheap.
  constexpr size_t kPayloadSize = 70 * 1024;
  std::string BigInsertion(kPayloadSize, 'x');

  // Set up the writer in an inner scope so its destructor flushes the .dia
  // file before we read it back.
  {
    FileSystemOptions FSOpts;
    FileManager FileMgr(FSOpts);
    DiagnosticOptions DiagOpts;
    DiagnosticsEngine Diags(DiagnosticIDs::create(), DiagOpts,
                            new IgnoringDiagConsumer());
    SourceManager SrcMgr(Diags, FileMgr);

    StringRef Path = "main.cpp";
    StringRef Contents = "int x;\n";
    FileEntryRef FE = FileMgr.getVirtualFileRef(
        Path, /*Size=*/static_cast<off_t>(Contents.size()),
        /*ModificationTime=*/0);
    SmallVector<char, 16> Buffer(Contents.begin(), Contents.end());
    SrcMgr.overrideFileContents(
        FE, std::make_unique<SmallVectorMemoryBuffer>(
                std::move(Buffer), Path, /*RequiresNullTerminator=*/false));
    FileID FID = SrcMgr.createFileID(FE, SourceLocation(), SrcMgr::C_User);
    SrcMgr.setMainFileID(FID);
    SourceLocation Loc = SrcMgr.translateLineCol(FID, 1, 1);

    // Attach the serialized-diagnostics consumer.
    Diags.setClient(serialized_diags::create(TmpPath, DiagOpts).release());
    Diags.setSourceManager(&SrcMgr);
    LangOptions LO;
    Diags.getClient()->BeginSourceFile(LO, /*PP=*/nullptr);

    // Emit one custom warning carrying the giant FixIt. Without the
    // RECORD_FIXIT VBR widening, this Report() asserts in
    // BitstreamWriter::Emit on +assertions builds.
    unsigned ID = Diags.getCustomDiagID(DiagnosticsEngine::Warning, "test");
    Diags.Report(Loc, ID) << FixItHint::CreateInsertion(Loc, BigInsertion);
  } // SDiagsWriter dtor flushes the .dia file here.

  // Read the file back and verify the FixIt round-trips intact.
  CapturingReader Reader;
  ASSERT_FALSE(Reader.readDiagnostics(TmpPath));
  ASSERT_EQ(Reader.FixItTexts.size(), 1u);
  EXPECT_EQ(Reader.FixItTexts[0].size(), kPayloadSize);
  EXPECT_EQ(Reader.FixItTexts[0], BigInsertion);
}

} // anonymous namespace
