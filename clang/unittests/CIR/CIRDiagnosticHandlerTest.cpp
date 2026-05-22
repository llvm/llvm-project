//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Unit tests for cir::CIRDiagnosticHandler.
//
// The handler routes MLIR-side diagnostics emitted during CIR passes, the
// verifier, and CIR-to-LLVM lowering through clang::DiagnosticsEngine. These
// tests exercise severity mapping, location translation, note attachment, and
// edge cases (missing files, fused/callsite/name locations, 0-line/0-col)
// without depending on a particular pass-side trigger.
//
//===----------------------------------------------------------------------===//

#include "../../lib/CIR/FrontendAction/CIRDiagnosticHandler.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"

#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/DiagnosticFrontend.h"
#include "clang/Basic/DiagnosticIDs.h"
#include "clang/Basic/DiagnosticOptions.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/SourceManager.h"

#include "llvm/Support/MemoryBuffer.h"
#include "gtest/gtest.h"

// Don't pull `using namespace clang;` or `using namespace mlir;` here:
// `clang::Diagnostic` and `mlir::Diagnostic` would clash.

namespace {

struct CapturedDiag {
  clang::DiagnosticsEngine::Level Level;
  unsigned ID;
  std::string Message;
  bool HasLoc;
  unsigned Line = 0;
  unsigned Column = 0;
};

class CapturingConsumer : public clang::DiagnosticConsumer {
public:
  std::vector<CapturedDiag> Diags;

  void HandleDiagnostic(clang::DiagnosticsEngine::Level Level,
                        const clang::Diagnostic &Info) override {
    clang::DiagnosticConsumer::HandleDiagnostic(Level, Info);
    CapturedDiag CD;
    CD.Level = Level;
    CD.ID = Info.getID();
    llvm::SmallString<128> Buf;
    Info.FormatDiagnostic(Buf);
    CD.Message = std::string(Buf);
    clang::SourceLocation Loc = Info.getLocation();
    CD.HasLoc = Loc.isValid();
    if (CD.HasLoc && Info.hasSourceManager()) {
      clang::PresumedLoc PL = Info.getSourceManager().getPresumedLoc(Loc);
      if (PL.isValid()) {
        CD.Line = PL.getLine();
        CD.Column = PL.getColumn();
      }
    }
    Diags.push_back(std::move(CD));
  }
};

class CIRDiagnosticHandlerTest : public ::testing::Test {
protected:
  CIRDiagnosticHandlerTest()
      : Consumer(new CapturingConsumer), FileMgr(FileMgrOpts),
        Diags(clang::DiagnosticIDs::create(), DiagOpts, Consumer.get(),
              /*ShouldOwnClient=*/false),
        SrcMgr(Diags, FileMgr) {
    // Register a virtual file so FileManager::getOptionalFileRef succeeds and
    // SourceManager has a buffer for line/col translation.
    constexpr llvm::StringLiteral Body =
        "line 1\nline 2\nline 3\nline 4\nline 5\n";
    auto Buf = llvm::MemoryBuffer::getMemBufferCopy(Body, srcName());
    clang::FileEntryRef File =
        FileMgr.getVirtualFileRef(srcName(), Buf->getBufferSize(), 0);
    SrcMgr.overrideFileContents(File, std::move(Buf));
    clang::FileID FID = SrcMgr.getOrCreateFileID(File, clang::SrcMgr::C_User);
    SrcMgr.setMainFileID(FID);
  }

  static llvm::StringRef srcName() { return "/virtual/file.c"; }

  mlir::Location fileLoc(unsigned Line, unsigned Col,
                         llvm::StringRef Path = srcName()) {
    return mlir::FileLineColLoc::get(&MLIRCtx, Path, Line, Col);
  }

  std::unique_ptr<CapturingConsumer> Consumer;
  clang::FileSystemOptions FileMgrOpts;
  clang::FileManager FileMgr;
  clang::DiagnosticOptions DiagOpts;
  clang::DiagnosticsEngine Diags;
  clang::SourceManager SrcMgr;
  mlir::MLIRContext MLIRCtx{mlir::MLIRContext::Threading::DISABLED};
};

TEST_F(CIRDiagnosticHandlerTest, ErrorRoutedWithSeverityAndLocation) {
  cir::CIRDiagnosticHandler Handler(&MLIRCtx, Diags, SrcMgr, FileMgr);
  mlir::emitError(fileLoc(2, 3)) << "boom";

  ASSERT_EQ(Consumer->Diags.size(), 1u);
  const CapturedDiag &D = Consumer->Diags.front();
  EXPECT_EQ(D.Level, clang::DiagnosticsEngine::Error);
  EXPECT_EQ(D.ID, static_cast<unsigned>(clang::diag::err_cir_mlir_diagnostic));
  EXPECT_EQ(D.Message, "boom");
  EXPECT_TRUE(D.HasLoc);
  EXPECT_EQ(D.Line, 2u);
  EXPECT_EQ(D.Column, 3u);
}

TEST_F(CIRDiagnosticHandlerTest, WarningRoutesToWarningId) {
  cir::CIRDiagnosticHandler Handler(&MLIRCtx, Diags, SrcMgr, FileMgr);
  mlir::emitWarning(fileLoc(1, 1)) << "soft";

  ASSERT_EQ(Consumer->Diags.size(), 1u);
  EXPECT_EQ(Consumer->Diags[0].Level, clang::DiagnosticsEngine::Warning);
  EXPECT_EQ(Consumer->Diags[0].ID,
            static_cast<unsigned>(clang::diag::warn_cir_mlir_diagnostic));
}

TEST_F(CIRDiagnosticHandlerTest, RemarkRoutesToRemarkId) {
  // Remarks are off by default. Promote -Rclangir so the consumer sees them.
  bool unknownGroup =
      Diags.setSeverityForGroup(clang::diag::Flavor::Remark, "remark-clangir",
                                clang::diag::Severity::Remark);
  ASSERT_FALSE(unknownGroup) << "remark-clangir group not registered";
  cir::CIRDiagnosticHandler Handler(&MLIRCtx, Diags, SrcMgr, FileMgr);
  mlir::emitRemark(fileLoc(1, 1)) << "fyi";

  ASSERT_EQ(Consumer->Diags.size(), 1u);
  EXPECT_EQ(Consumer->Diags[0].Level, clang::DiagnosticsEngine::Remark);
  EXPECT_EQ(Consumer->Diags[0].ID,
            static_cast<unsigned>(clang::diag::remark_cir_mlir_diagnostic));
}

TEST_F(CIRDiagnosticHandlerTest, NotesAttachAfterPrimary) {
  cir::CIRDiagnosticHandler Handler(&MLIRCtx, Diags, SrcMgr, FileMgr);
  {
    mlir::InFlightDiagnostic d = mlir::emitError(fileLoc(3, 4)) << "primary";
    d.attachNote(fileLoc(4, 5)) << "extra info";
  } // InFlightDiagnostic dtor reports.

  ASSERT_EQ(Consumer->Diags.size(), 2u);
  EXPECT_EQ(Consumer->Diags[0].Level, clang::DiagnosticsEngine::Error);
  EXPECT_EQ(Consumer->Diags[0].ID,
            static_cast<unsigned>(clang::diag::err_cir_mlir_diagnostic));
  EXPECT_EQ(Consumer->Diags[1].Level, clang::DiagnosticsEngine::Note);
  EXPECT_EQ(Consumer->Diags[1].ID,
            static_cast<unsigned>(clang::diag::note_cir_mlir_diagnostic));
  EXPECT_EQ(Consumer->Diags[1].Message, "extra info");
  EXPECT_EQ(Consumer->Diags[1].Line, 4u);
  EXPECT_EQ(Consumer->Diags[1].Column, 5u);
}

TEST_F(CIRDiagnosticHandlerTest, ZeroLineColumnFallsBackToInvalidLoc) {
  // Module-level locations use line 0 / column 0; SourceManager would assert.
  // The handler must guard and emit without a source location.
  cir::CIRDiagnosticHandler Handler(&MLIRCtx, Diags, SrcMgr, FileMgr);
  mlir::emitError(fileLoc(0, 0)) << "module-level";

  ASSERT_EQ(Consumer->Diags.size(), 1u);
  EXPECT_FALSE(Consumer->Diags[0].HasLoc);
  EXPECT_EQ(Consumer->Diags[0].Message, "module-level");
}

TEST_F(CIRDiagnosticHandlerTest, FileNotInManagerFallsBack) {
  cir::CIRDiagnosticHandler Handler(&MLIRCtx, Diags, SrcMgr, FileMgr);
  mlir::emitError(fileLoc(1, 1, "/not/registered.c")) << "stray";

  ASSERT_EQ(Consumer->Diags.size(), 1u);
  EXPECT_FALSE(Consumer->Diags[0].HasLoc);
  EXPECT_EQ(Consumer->Diags[0].Message, "stray");
}

TEST_F(CIRDiagnosticHandlerTest, FusedLocResolvesToFirstTranslatableChild) {
  cir::CIRDiagnosticHandler Handler(&MLIRCtx, Diags, SrcMgr, FileMgr);
  mlir::Location unknown = mlir::UnknownLoc::get(&MLIRCtx);
  mlir::Location good = fileLoc(2, 1);
  mlir::Location fused = mlir::FusedLoc::get({unknown, good}, {}, &MLIRCtx);
  mlir::emitError(fused) << "fused";

  ASSERT_EQ(Consumer->Diags.size(), 1u);
  EXPECT_TRUE(Consumer->Diags[0].HasLoc);
  EXPECT_EQ(Consumer->Diags[0].Line, 2u);
}

TEST_F(CIRDiagnosticHandlerTest, CallSiteLocFollowsCallee) {
  cir::CIRDiagnosticHandler Handler(&MLIRCtx, Diags, SrcMgr, FileMgr);
  mlir::Location callee = fileLoc(3, 2);
  mlir::Location caller = fileLoc(5, 1);
  mlir::Location cs = mlir::CallSiteLoc::get(callee, caller);
  mlir::emitError(cs) << "cs";

  ASSERT_EQ(Consumer->Diags.size(), 1u);
  EXPECT_EQ(Consumer->Diags[0].Line, 3u);
  EXPECT_EQ(Consumer->Diags[0].Column, 2u);
}

TEST_F(CIRDiagnosticHandlerTest, NameLocFollowsChild) {
  cir::CIRDiagnosticHandler Handler(&MLIRCtx, Diags, SrcMgr, FileMgr);
  mlir::Location child = fileLoc(4, 1);
  mlir::Location named =
      mlir::NameLoc::get(mlir::StringAttr::get(&MLIRCtx, "tag"), child);
  mlir::emitError(named) << "named";

  ASSERT_EQ(Consumer->Diags.size(), 1u);
  EXPECT_EQ(Consumer->Diags[0].Line, 4u);
}

TEST_F(CIRDiagnosticHandlerTest, UnknownLocEmitsUnattached) {
  cir::CIRDiagnosticHandler Handler(&MLIRCtx, Diags, SrcMgr, FileMgr);
  mlir::emitError(mlir::UnknownLoc::get(&MLIRCtx)) << "lost";

  ASSERT_EQ(Consumer->Diags.size(), 1u);
  EXPECT_FALSE(Consumer->Diags[0].HasLoc);
}

TEST_F(CIRDiagnosticHandlerTest, ScopedLifetimeUnregistersOnDestruction) {
  // After the handler scope ends, MLIR's default handler takes over.
  // Verify the consumer no longer receives diagnostics through our path.
  {
    cir::CIRDiagnosticHandler Handler(&MLIRCtx, Diags, SrcMgr, FileMgr);
    mlir::emitError(fileLoc(1, 1)) << "in-scope";
  }
  size_t scopedCount = Consumer->Diags.size();
  mlir::emitError(fileLoc(1, 1)) << "out-of-scope";
  // Default handler prints to stderr but does not call our consumer.
  EXPECT_EQ(Consumer->Diags.size(), scopedCount);
}

} // namespace
