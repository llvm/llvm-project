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
#include "gtest/gtest.h"

using namespace llvm;
using namespace clang;

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

// Check that SuppressAfterFatalError works as intended
TEST(DiagnosticTest, suppressAfterFatalError) {
  for (unsigned Suppress = 0; Suppress != 2; ++Suppress) {
    DiagnosticsEngine Diags(new DiagnosticIDs(),
                            new DiagnosticOptions,
                            new IgnoringDiagConsumer());
    Diags.setSuppressAfterFatalError(Suppress);

    // Diag that would set UnrecoverableErrorOccurred and ErrorOccurred.
    Diags.Report(diag::err_cannot_open_file) << "file" << "error";

    // Diag that would set FatalErrorOccurred
    // (via non-note following a fatal error).
    Diags.Report(diag::warn_mt_message) << "warning";

    EXPECT_TRUE(Diags.hasErrorOccurred());
    EXPECT_TRUE(Diags.hasFatalErrorOccurred());
    EXPECT_TRUE(Diags.hasUncompilableErrorOccurred());
    EXPECT_TRUE(Diags.hasUnrecoverableErrorOccurred());

    // The warning should be emitted and counted only if we're not suppressing
    // after fatal errors.
    EXPECT_EQ(Diags.getNumWarnings(), Suppress ? 0u : 1u);
  }
}

TEST(DiagnosticTest, diagnosticError) {
  DiagnosticsEngine Diags(new DiagnosticIDs(), new DiagnosticOptions,
                          new IgnoringDiagConsumer());
  PartialDiagnostic::StorageAllocator Alloc;
  llvm::Expected<std::pair<int, int>> Value = DiagnosticError::create(
      SourceLocation(), PartialDiagnostic(diag::err_cannot_open_file, Alloc)
                            << "file"
                            << "error");
  ASSERT_TRUE(!Value);
  llvm::Error Err = Value.takeError();
  Optional<PartialDiagnosticAt> ErrDiag = DiagnosticError::take(Err);
  llvm::cantFail(std::move(Err));
  ASSERT_FALSE(!ErrDiag);
  EXPECT_EQ(ErrDiag->first, SourceLocation());
  EXPECT_EQ(ErrDiag->second.getDiagID(), diag::err_cannot_open_file);

  Value = std::make_pair(20, 1);
  ASSERT_FALSE(!Value);
  EXPECT_EQ(*Value, std::make_pair(20, 1));
  EXPECT_EQ(Value->first, 20);
}
}
