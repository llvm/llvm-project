//===- unittests/Basic/DiagnosticTest.cpp -- Diagnostic engine tests ------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "clang/Basic/Diagnostic.h"
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

// Check that FatalsAsErrors works as intended
TEST(DiagnosticTest, fatalsAsErrors) {
  DiagnosticsEngine Diags(new DiagnosticIDs(),
                          new DiagnosticOptions,
                          new IgnoringDiagConsumer());
  Diags.setFatalsAsError(true);

  // Diag that would set UncompilableErrorOccurred and ErrorOccurred.
  Diags.Report(diag::err_target_unknown_triple) << "unknown";

  // Diag that would set UnrecoverableErrorOccurred and ErrorOccurred.
  Diags.Report(diag::err_cannot_open_file) << "file" << "error";

  // Diag that would set FatalErrorOccurred
  // (via non-note following a fatal error).
  Diags.Report(diag::warn_mt_message) << "warning";

  EXPECT_TRUE(Diags.hasErrorOccurred());
  EXPECT_FALSE(Diags.hasFatalErrorOccurred());
  EXPECT_TRUE(Diags.hasUncompilableErrorOccurred());
  EXPECT_TRUE(Diags.hasUnrecoverableErrorOccurred());
}

}
