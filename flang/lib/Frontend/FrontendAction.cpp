//===--- FrontendAction.cpp -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Frontend/FrontendAction.h"
#include "flang/Frontend/CompilerInstance.h"
#include "flang/Frontend/FrontendActions.h"
#include "flang/Frontend/FrontendOptions.h"
#include "flang/FrontendTool/Utils.h"
#include "clang/Basic/DiagnosticFrontend.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/VirtualFileSystem.h"

using namespace Fortran::frontend;

void FrontendAction::set_currentInput(const FrontendInputFile &currentInput) {
  this->currentInput_ = currentInput;
}

// Call this method if BeginSourceFile fails.
// Deallocate compiler instance, input and output descriptors
static void BeginSourceFileCleanUp(FrontendAction &fa, CompilerInstance &ci) {
  ci.ClearOutputFiles(/*EraseFiles=*/true);
  fa.set_currentInput(FrontendInputFile());
  fa.set_instance(nullptr);
}

bool FrontendAction::BeginSourceFile(
    CompilerInstance &ci, const FrontendInputFile &realInput) {

  FrontendInputFile input(realInput);

  // Return immediately if the input file does not exist or is not a file. Note
  // that we cannot check this for input from stdin.
  if (input.file() != "-") {
    if (!llvm::sys::fs::is_regular_file(input.file())) {
      // Create an diagnostic ID to report
      unsigned diagID;
      if (llvm::vfs::getRealFileSystem()->exists(input.file())) {
        ci.diagnostics().Report(clang::diag::err_fe_error_reading)
            << input.file();
        diagID = ci.diagnostics().getCustomDiagID(
            clang::DiagnosticsEngine::Error, "%0 is not a regular file");
      } else {
        diagID = ci.diagnostics().getCustomDiagID(
            clang::DiagnosticsEngine::Error, "%0 does not exist");
      }

      // Report the diagnostic and return
      ci.diagnostics().Report(diagID) << input.file();
      BeginSourceFileCleanUp(*this, ci);
      return false;
    }
  }

  assert(!instance_ && "Already processing a source file!");
  assert(!realInput.IsEmpty() && "Unexpected empty filename!");
  set_currentInput(realInput);
  set_instance(&ci);

  if (!ci.HasAllSources()) {
    BeginSourceFileCleanUp(*this, ci);
    return false;
  }

  if (!BeginSourceFileAction(ci)) {
    BeginSourceFileCleanUp(*this, ci);
    return false;
  }

  return true;
}

bool FrontendAction::ShouldEraseOutputFiles() {
  return instance().diagnostics().hasErrorOccurred();
}

llvm::Error FrontendAction::Execute() {
  ExecuteAction();

  return llvm::Error::success();
}

void FrontendAction::EndSourceFile() {
  CompilerInstance &ci = instance();

  // Cleanup the output streams, and erase the output files if instructed by the
  // FrontendAction.
  ci.ClearOutputFiles(/*EraseFiles=*/ShouldEraseOutputFiles());

  set_instance(nullptr);
  set_currentInput(FrontendInputFile());
}
