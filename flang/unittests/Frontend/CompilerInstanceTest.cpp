//===- unittests/Frontend/CompilerInstanceTest.cpp - CI tests -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Frontend/CompilerInstance.h"
#include "clang/Basic/DiagnosticOptions.h"
#include "clang/Frontend/TextDiagnosticPrinter.h"

#include "gtest/gtest.h"

using namespace llvm;
using namespace Fortran::frontend;

namespace {

TEST(CompilerInstance, AllowDiagnosticLogWithUnownedDiagnosticConsumer) {
  // 1. Set-up a basic DiagnosticConsumer
  std::string diagnosticOutput;
  llvm::raw_string_ostream diagnosticsOS(diagnosticOutput);
  auto diagPrinter = std::make_unique<clang::TextDiagnosticPrinter>(
      diagnosticsOS, new clang::DiagnosticOptions());

  // 2. Create a CompilerInstance (to manage a DiagnosticEngine)
  CompilerInstance compInst;

  // 3. Set-up DiagnosticOptions
  auto diagOpts = new clang::DiagnosticOptions();
  // Tell the diagnostics engine to emit the diagnostic log to STDERR. This
  // ensures that a chained diagnostic consumer is created so that the test can
  // exercise the unowned diagnostic consumer in a chained consumer.
  diagOpts->DiagnosticLogFile = "-";

  // 4. Create a DiagnosticEngine with an unowned consumer
  IntrusiveRefCntPtr<clang::DiagnosticsEngine> diags =
      compInst.CreateDiagnostics(diagOpts, diagPrinter.get(),
          /*ShouldOwnClient=*/false);

  // 5. Report a diagnostic
  diags->Report(clang::diag::err_expected) << "no crash";

  // 6. Verify that the reported diagnostic wasn't lost and did end up in the
  // output stream
  ASSERT_EQ(diagnosticsOS.str(), "error: expected no crash\n");
}
} // namespace
