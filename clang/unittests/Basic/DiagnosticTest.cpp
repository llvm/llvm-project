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
#include "gtest/gtest.h"
#include <optional>

using namespace llvm;
using namespace clang;

void clang::DiagnosticsTestHelper(DiagnosticsEngine &diag) {
  unsigned delayedDiagID = 0U;

  EXPECT_EQ(diag.DelayedDiagID, delayedDiagID);
  EXPECT_FALSE(diag.DiagStates.empty());
  EXPECT_TRUE(diag.DiagStatesByLoc.empty());
  EXPECT_TRUE(diag.DiagStateOnPushStack.empty());
}

namespace {

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

  EXPECT_FALSE(Diags.isDiagnosticInFlight());
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
}
