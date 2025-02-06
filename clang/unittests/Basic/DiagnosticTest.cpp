//===- unittests/Basic/DiagnosticTest.cpp -- Diagnostic engine tests ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/DiagnosticError.h"
#include "clang/Basic/DiagnosticIDs.h"
#include "clang/Basic/DiagnosticLex.h"
#include "clang/Basic/DiagnosticSema.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/SourceManager.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/IntrusiveRefCntPtr.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/VirtualFileSystem.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <memory>
#include <optional>
#include <vector>

using namespace llvm;
using namespace clang;

// Declare DiagnosticsTestHelper to avoid GCC warning
namespace clang {
void DiagnosticsTestHelper(DiagnosticsEngine &diag);
}

void clang::DiagnosticsTestHelper(DiagnosticsEngine &diag) {
  EXPECT_FALSE(diag.DiagStates.empty());
  EXPECT_TRUE(diag.DiagStatesByLoc.empty());
  EXPECT_TRUE(diag.DiagStateOnPushStack.empty());
}

namespace {
using testing::AllOf;
using testing::ElementsAre;
using testing::IsEmpty;

// Check that DiagnosticErrorTrap works with SuppressAllDiagnostics.
TEST(DiagnosticTest, suppressAndTrap) {
  DiagnosticsEngine Diags(new DiagnosticIDs(),
                          new DiagnosticOptions,
                          new IgnoringDiagConsumer());
  Diags.setSuppressAllDiagnostics(true);

  {
    DiagnosticErrorTrap trap(Diags);

    // Diag that would set UncompilableErrorOccurred and ErrorOccurred.
    Diags.Report(diag::err_target_unknown_triple) << "unknown";

    // Diag that would set UnrecoverableErrorOccurred and ErrorOccurred.
    Diags.Report(diag::err_cannot_open_file) << "file" << "error";

    // Diag that would set FatalErrorOccurred
    // (via non-note following a fatal error).
    Diags.Report(diag::warn_mt_message) << "warning";

    EXPECT_TRUE(trap.hasErrorOccurred());
    EXPECT_TRUE(trap.hasUnrecoverableErrorOccurred());
  }

  EXPECT_FALSE(Diags.hasErrorOccurred());
  EXPECT_FALSE(Diags.hasFatalErrorOccurred());
  EXPECT_FALSE(Diags.hasUncompilableErrorOccurred());
  EXPECT_FALSE(Diags.hasUnrecoverableErrorOccurred());
}

// Check that FatalsAsError works as intended
TEST(DiagnosticTest, fatalsAsError) {
  for (unsigned FatalsAsError = 0; FatalsAsError != 2; ++FatalsAsError) {
    DiagnosticsEngine Diags(new DiagnosticIDs(),
                            new DiagnosticOptions,
                            new IgnoringDiagConsumer());
    Diags.setFatalsAsError(FatalsAsError);

    // Diag that would set UnrecoverableErrorOccurred and ErrorOccurred.
    Diags.Report(diag::err_cannot_open_file) << "file" << "error";

    // Diag that would set FatalErrorOccurred
    // (via non-note following a fatal error).
    Diags.Report(diag::warn_mt_message) << "warning";

    EXPECT_TRUE(Diags.hasErrorOccurred());
    EXPECT_EQ(Diags.hasFatalErrorOccurred(), FatalsAsError ? 0u : 1u);
    EXPECT_TRUE(Diags.hasUncompilableErrorOccurred());
    EXPECT_TRUE(Diags.hasUnrecoverableErrorOccurred());

    // The warning should be emitted and counted only if we're not suppressing
    // after fatal errors.
    EXPECT_EQ(Diags.getNumWarnings(), FatalsAsError);
  }
}

TEST(DiagnosticTest, tooManyErrorsIsAlwaysFatal) {
  DiagnosticsEngine Diags(new DiagnosticIDs(), new DiagnosticOptions,
                          new IgnoringDiagConsumer());
  Diags.setFatalsAsError(true);

  // Report a fatal_too_many_errors diagnostic to ensure that still
  // acts as a fatal error despite downgrading fatal errors to errors.
  Diags.Report(diag::fatal_too_many_errors);
  EXPECT_TRUE(Diags.hasFatalErrorOccurred());

  // Ensure that the severity of that diagnostic is really "fatal".
  EXPECT_EQ(Diags.getDiagnosticLevel(diag::fatal_too_many_errors, {}),
            DiagnosticsEngine::Level::Fatal);
}

// Check that soft RESET works as intended
TEST(DiagnosticTest, softReset) {
  DiagnosticsEngine Diags(new DiagnosticIDs(), new DiagnosticOptions,
                          new IgnoringDiagConsumer());

  unsigned numWarnings = 0U, numErrors = 0U;

  Diags.Reset(true);
  // Check For ErrorOccurred and TrapNumErrorsOccurred
  EXPECT_FALSE(Diags.hasErrorOccurred());
  EXPECT_FALSE(Diags.hasFatalErrorOccurred());
  EXPECT_FALSE(Diags.hasUncompilableErrorOccurred());
  // Check for UnrecoverableErrorOccurred and TrapNumUnrecoverableErrorsOccurred
  EXPECT_FALSE(Diags.hasUnrecoverableErrorOccurred());

  EXPECT_EQ(Diags.getNumWarnings(), numWarnings);
  EXPECT_EQ(Diags.getNumErrors(), numErrors);

  // Check for private variables of DiagnosticsEngine differentiating soft reset
  DiagnosticsTestHelper(Diags);

  EXPECT_TRUE(Diags.isLastDiagnosticIgnored());
}

TEST(DiagnosticTest, diagnosticError) {
  DiagnosticsEngine Diags(new DiagnosticIDs(), new DiagnosticOptions,
                          new IgnoringDiagConsumer());
  PartialDiagnostic::DiagStorageAllocator Alloc;
  llvm::Expected<std::pair<int, int>> Value = DiagnosticError::create(
      SourceLocation(), PartialDiagnostic(diag::err_cannot_open_file, Alloc)
                            << "file"
                            << "error");
  ASSERT_TRUE(!Value);
  llvm::Error Err = Value.takeError();
  std::optional<PartialDiagnosticAt> ErrDiag = DiagnosticError::take(Err);
  llvm::cantFail(std::move(Err));
  ASSERT_FALSE(!ErrDiag);
  EXPECT_EQ(ErrDiag->first, SourceLocation());
  EXPECT_EQ(ErrDiag->second.getDiagID(), diag::err_cannot_open_file);

  Value = std::make_pair(20, 1);
  ASSERT_FALSE(!Value);
  EXPECT_EQ(*Value, std::make_pair(20, 1));
  EXPECT_EQ(Value->first, 20);
}

TEST(DiagnosticTest, storedDiagEmptyWarning) {
  DiagnosticsEngine Diags(new DiagnosticIDs(), new DiagnosticOptions);

  class CaptureDiagnosticConsumer : public DiagnosticConsumer {
  public:
    SmallVector<StoredDiagnostic> StoredDiags;

    void HandleDiagnostic(DiagnosticsEngine::Level level,
                          const Diagnostic &Info) override {
      StoredDiags.push_back(StoredDiagnostic(level, Info));
    }
  };

  CaptureDiagnosticConsumer CaptureConsumer;
  Diags.setClient(&CaptureConsumer, /*ShouldOwnClient=*/false);
  Diags.Report(diag::pp_hash_warning) << "";
  ASSERT_TRUE(CaptureConsumer.StoredDiags.size() == 1);

  // Make sure an empty warning can round-trip with \c StoredDiagnostic.
  Diags.Report(CaptureConsumer.StoredDiags.front());
}

class SuppressionMappingTest : public testing::Test {
public:
  SuppressionMappingTest() {
    Diags.setClient(&CaptureConsumer, /*ShouldOwnClient=*/false);
  }

protected:
  llvm::IntrusiveRefCntPtr<llvm::vfs::InMemoryFileSystem> FS =
      llvm::makeIntrusiveRefCnt<llvm::vfs::InMemoryFileSystem>();
  DiagnosticsEngine Diags{new DiagnosticIDs(), new DiagnosticOptions};

  llvm::ArrayRef<StoredDiagnostic> diags() {
    return CaptureConsumer.StoredDiags;
  }

  SourceLocation locForFile(llvm::StringRef FileName) {
    auto Buf = MemoryBuffer::getMemBuffer("", FileName);
    SourceManager &SM = Diags.getSourceManager();
    FileID FooID = SM.createFileID(std::move(Buf));
    return SM.getLocForStartOfFile(FooID);
  }

private:
  FileManager FM{{}, FS};
  SourceManager SM{Diags, FM};

  class CaptureDiagnosticConsumer : public DiagnosticConsumer {
  public:
    std::vector<StoredDiagnostic> StoredDiags;

    void HandleDiagnostic(DiagnosticsEngine::Level level,
                          const Diagnostic &Info) override {
      StoredDiags.push_back(StoredDiagnostic(level, Info));
    }
  };
  CaptureDiagnosticConsumer CaptureConsumer;
};

MATCHER_P(WithMessage, Msg, "has diagnostic message") {
  return arg.getMessage() == Msg;
}
MATCHER(IsError, "has error severity") {
  return arg.getLevel() == DiagnosticsEngine::Level::Error;
}

TEST_F(SuppressionMappingTest, MissingMappingFile) {
  Diags.getDiagnosticOptions().DiagnosticSuppressionMappingsFile = "foo.txt";
  clang::ProcessWarningOptions(Diags, Diags.getDiagnosticOptions(), *FS);
  EXPECT_THAT(diags(), ElementsAre(AllOf(
                           WithMessage("no such file or directory: 'foo.txt'"),
                           IsError())));
}

TEST_F(SuppressionMappingTest, MalformedFile) {
  Diags.getDiagnosticOptions().DiagnosticSuppressionMappingsFile = "foo.txt";
  FS->addFile("foo.txt", /*ModificationTime=*/{},
              llvm::MemoryBuffer::getMemBuffer("asdf", "foo.txt"));
  clang::ProcessWarningOptions(Diags, Diags.getDiagnosticOptions(), *FS);
  EXPECT_THAT(diags(),
              ElementsAre(AllOf(
                  WithMessage("failed to process suppression mapping file "
                              "'foo.txt': malformed line 1: 'asdf'"),
                  IsError())));
}

TEST_F(SuppressionMappingTest, UnknownDiagName) {
  Diags.getDiagnosticOptions().DiagnosticSuppressionMappingsFile = "foo.txt";
  FS->addFile("foo.txt", /*ModificationTime=*/{},
              llvm::MemoryBuffer::getMemBuffer("[non-existing-warning]"));
  clang::ProcessWarningOptions(Diags, Diags.getDiagnosticOptions(), *FS);
  EXPECT_THAT(diags(), ElementsAre(WithMessage(
                           "unknown warning option 'non-existing-warning'")));
}

TEST_F(SuppressionMappingTest, SuppressesGroup) {
  llvm::StringLiteral SuppressionMappingFile = R"(
  [unused]
  src:*)";
  Diags.getDiagnosticOptions().DiagnosticSuppressionMappingsFile = "foo.txt";
  FS->addFile("foo.txt", /*ModificationTime=*/{},
              llvm::MemoryBuffer::getMemBuffer(SuppressionMappingFile));
  clang::ProcessWarningOptions(Diags, Diags.getDiagnosticOptions(), *FS);
  EXPECT_THAT(diags(), IsEmpty());

  SourceLocation FooLoc = locForFile("foo.cpp");
  EXPECT_TRUE(Diags.isSuppressedViaMapping(diag::warn_unused_function, FooLoc));
  EXPECT_FALSE(Diags.isSuppressedViaMapping(diag::warn_deprecated, FooLoc));
}

TEST_F(SuppressionMappingTest, EmitCategoryIsExcluded) {
  llvm::StringLiteral SuppressionMappingFile = R"(
  [unused]
  src:*
  src:*foo.cpp=emit)";
  Diags.getDiagnosticOptions().DiagnosticSuppressionMappingsFile = "foo.txt";
  FS->addFile("foo.txt", /*ModificationTime=*/{},
              llvm::MemoryBuffer::getMemBuffer(SuppressionMappingFile));
  clang::ProcessWarningOptions(Diags, Diags.getDiagnosticOptions(), *FS);
  EXPECT_THAT(diags(), IsEmpty());

  EXPECT_TRUE(Diags.isSuppressedViaMapping(diag::warn_unused_function,
                                           locForFile("bar.cpp")));
  EXPECT_FALSE(Diags.isSuppressedViaMapping(diag::warn_unused_function,
                                            locForFile("foo.cpp")));
}

TEST_F(SuppressionMappingTest, LongestMatchWins) {
  llvm::StringLiteral SuppressionMappingFile = R"(
  [unused]
  src:*clang/*
  src:*clang/lib/Sema/*=emit
  src:*clang/lib/Sema/foo*)";
  Diags.getDiagnosticOptions().DiagnosticSuppressionMappingsFile = "foo.txt";
  FS->addFile("foo.txt", /*ModificationTime=*/{},
              llvm::MemoryBuffer::getMemBuffer(SuppressionMappingFile));
  clang::ProcessWarningOptions(Diags, Diags.getDiagnosticOptions(), *FS);
  EXPECT_THAT(diags(), IsEmpty());

  EXPECT_TRUE(Diags.isSuppressedViaMapping(
      diag::warn_unused_function, locForFile("clang/lib/Basic/foo.h")));
  EXPECT_FALSE(Diags.isSuppressedViaMapping(
      diag::warn_unused_function, locForFile("clang/lib/Sema/bar.h")));
  EXPECT_TRUE(Diags.isSuppressedViaMapping(diag::warn_unused_function,
                                           locForFile("clang/lib/Sema/foo.h")));
}

TEST_F(SuppressionMappingTest, IsIgnored) {
  llvm::StringLiteral SuppressionMappingFile = R"(
  [unused]
  src:*clang/*)";
  Diags.getDiagnosticOptions().DiagnosticSuppressionMappingsFile = "foo.txt";
  Diags.getDiagnosticOptions().Warnings = {"unused"};
  FS->addFile("foo.txt", /*ModificationTime=*/{},
              llvm::MemoryBuffer::getMemBuffer(SuppressionMappingFile));
  clang::ProcessWarningOptions(Diags, Diags.getDiagnosticOptions(), *FS);
  ASSERT_THAT(diags(), IsEmpty());

  SourceManager &SM = Diags.getSourceManager();
  auto ClangID =
      SM.createFileID(llvm::MemoryBuffer::getMemBuffer("", "clang/foo.h"));
  auto NonClangID =
      SM.createFileID(llvm::MemoryBuffer::getMemBuffer("", "llvm/foo.h"));
  auto PresumedClangID =
      SM.createFileID(llvm::MemoryBuffer::getMemBuffer("", "llvm/foo2.h"));
  // Add a line directive to point into clang/foo.h
  SM.AddLineNote(SM.getLocForStartOfFile(PresumedClangID), 42,
                 SM.getLineTableFilenameID("clang/foo.h"), false, false,
                 clang::SrcMgr::C_User);

  EXPECT_TRUE(Diags.isIgnored(diag::warn_unused_function,
                              SM.getLocForStartOfFile(ClangID)));
  EXPECT_FALSE(Diags.isIgnored(diag::warn_unused_function,
                               SM.getLocForStartOfFile(NonClangID)));
  EXPECT_TRUE(Diags.isIgnored(diag::warn_unused_function,
                              SM.getLocForStartOfFile(PresumedClangID)));

  // Pretend we have a clang-diagnostic pragma to enforce the warning. Make sure
  // suppressing mapping doesn't take over.
  Diags.setSeverity(diag::warn_unused_function, diag::Severity::Error,
                    SM.getLocForStartOfFile(ClangID));
  EXPECT_FALSE(Diags.isIgnored(diag::warn_unused_function,
                               SM.getLocForStartOfFile(ClangID)));
}

TEST_F(SuppressionMappingTest, ParsingRespectsOtherWarningOpts) {
  Diags.getDiagnosticOptions().DiagnosticSuppressionMappingsFile = "foo.txt";
  FS->addFile("foo.txt", /*ModificationTime=*/{},
              llvm::MemoryBuffer::getMemBuffer("[non-existing-warning]"));
  Diags.getDiagnosticOptions().Warnings.push_back("no-unknown-warning-option");
  clang::ProcessWarningOptions(Diags, Diags.getDiagnosticOptions(), *FS);
  EXPECT_THAT(diags(), IsEmpty());
}
} // namespace
