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
#include "llvm/Support/Errc.h"

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
  assert(!instance_ && "Already processing a source file!");
  assert(!realInput.IsEmpty() && "Unexpected empty filename!");
  set_currentInput(realInput);
  set_instance(&ci);
  if (!ci.HasAllSources()) {
    BeginSourceFileCleanUp(*this, ci);
    return false;
  }
  return true;
}

bool FrontendAction::ShouldEraseOutputFiles() {
  return instance().diagnostics().hasErrorOccurred();
}

llvm::Error FrontendAction::Execute() {
  CompilerInstance &ci = this->instance();

  std::string currentInputPath{GetCurrentFileOrBufferName()};

  Fortran::parser::Options parserOptions =
      this->instance().invocation().fortranOpts();

  // Prescan. In case of failure, report and return.
  ci.parsing().Prescan(currentInputPath, parserOptions);

  if (ci.parsing().messages().AnyFatalError()) {
    const unsigned diagID = ci.diagnostics().getCustomDiagID(
        clang::DiagnosticsEngine::Error, "Could not scan %0");
    ci.diagnostics().Report(diagID) << GetCurrentFileOrBufferName();
    ci.parsing().messages().Emit(llvm::errs(), ci.allCookedSources());

    return llvm::Error::success();
  }

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
