//===- unittests/Frontend/CompilerInstanceTest.cpp - CI tests -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Frontend/CompilerInstance.h"
#include "flang/Frontend/TextDiagnosticPrinter.h"
#include "clang/Basic/DiagnosticOptions.h"
#include "llvm/Support/FileSystem.h"

#include "gtest/gtest.h"

using namespace llvm;
using namespace Fortran::frontend;

namespace {

TEST(CompilerInstance, SanityCheckForFileManager) {
  const char *inputSource = "InputSourceFile";
  std::string inputFile = "buffer-file-test.f";
  std::error_code ec;

  // 1. Create the input file for the file manager
  // AllSources (which is used to manage files inside every compiler instance),
  // works with paths. This means that it requires a physical file. Create one.
  std::unique_ptr<llvm::raw_fd_ostream> os{
      new llvm::raw_fd_ostream(inputFile, ec, llvm::sys::fs::OF_None)};
  if (ec)
    FAIL() << "Failed to create the input file";

  // Populate the input file with the pre-defined input and flush it.
  *(os) << inputSource;
  os.reset();

  // Get the path of the input file
  llvm::SmallString<64> cwd;
  if (std::error_code ec = llvm::sys::fs::current_path(cwd))
    FAIL() << "Failed to obtain the current working directory";
  std::string testFilePath(cwd.c_str());
  testFilePath += "/" + inputFile;

  // 2. Set up CompilerInstance (i.e. specify the input file)
  std::string buf;
  llvm::raw_string_ostream errorStream{buf};
  CompilerInstance compInst;
  const Fortran::parser::SourceFile *sf =
      compInst.getAllSources().Open(testFilePath, errorStream);

  // 3. Verify the content of the input file
  // This is just a sanity check to make sure that CompilerInstance is capable
  // of reading input files.
  llvm::ArrayRef<char> fileContent = sf->content();
  EXPECT_FALSE(fileContent.size() == 0);
  EXPECT_TRUE(
      llvm::StringRef(fileContent.data()).starts_with("InputSourceFile"));

  // 4. Delete the test file
  ec = llvm::sys::fs::remove(inputFile);
  if (ec)
    FAIL() << "Failed to delete the test file";
}

TEST(CompilerInstance, AllowDiagnosticLogWithUnownedDiagnosticConsumer) {
  // 1. Set-up a basic DiagnosticConsumer
  std::string diagnosticOutput;
  llvm::raw_string_ostream diagnosticsOS(diagnosticOutput);
  clang::DiagnosticOptions diagPrinterOpts;
  auto diagPrinter = std::make_unique<Fortran::frontend::TextDiagnosticPrinter>(
      diagnosticsOS, diagPrinterOpts);

  // 2. Create a CompilerInstance (to manage a DiagnosticEngine)
  CompilerInstance compInst;

  // 3. Set-up DiagnosticOptions
  clang::DiagnosticOptions diagOpts;
  // Tell the diagnostics engine to emit the diagnostic log to STDERR. This
  // ensures that a chained diagnostic consumer is created so that the test can
  // exercise the unowned diagnostic consumer in a chained consumer.
  diagOpts.DiagnosticLogFile = "-";

  // 4. Create a DiagnosticEngine with an unowned consumer
  IntrusiveRefCntPtr<clang::DiagnosticsEngine> diags =
      compInst.createDiagnostics(diagOpts, diagPrinter.get(),
          /*ShouldOwnClient=*/false);

  // 5. Report a diagnostic
  diags->Report(clang::diag::err_expected) << "no crash";

  // 6. Verify that the reported diagnostic wasn't lost and did end up in the
  // output stream
  ASSERT_EQ(diagnosticOutput, "error: expected no crash\n");
}
} // namespace
