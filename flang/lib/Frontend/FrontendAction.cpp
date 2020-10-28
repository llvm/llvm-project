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

void FrontendAction::SetCurrentInput(const FrontendInputFile &currentInput) {
  this->currentInput_ = currentInput;
}

// Call this method if BeginSourceFile fails.
// Deallocate compiler instance, input and output descriptors
static void BeginSourceFileCleanUp(FrontendAction &fa, CompilerInstance &ci) {
  ci.ClearOutputFiles(/*EraseFiles=*/true);
  fa.SetCurrentInput(FrontendInputFile());
  fa.SetCompilerInstance(nullptr);
}

bool FrontendAction::BeginSourceFile(
    CompilerInstance &ci, const FrontendInputFile &realInput) {

  FrontendInputFile input(realInput);
  assert(!instance_ && "Already processing a source file!");
  assert(!realInput.IsEmpty() && "Unexpected empty filename!");
  SetCurrentInput(realInput);
  SetCompilerInstance(&ci);
  if (!ci.HasAllSources()) {
    BeginSourceFileCleanUp(*this, ci);
    return false;
  }
  return true;
}

bool FrontendAction::ShouldEraseOutputFiles() {
  return GetCompilerInstance().getDiagnostics().hasErrorOccurred();
}

llvm::Error FrontendAction::Execute() {
  ExecuteAction();
  return llvm::Error::success();
}

void FrontendAction::EndSourceFile() {
  CompilerInstance &ci = GetCompilerInstance();

  // Cleanup the output streams, and erase the output files if instructed by the
  // FrontendAction.
  ci.ClearOutputFiles(/*EraseFiles=*/ShouldEraseOutputFiles());

  SetCompilerInstance(nullptr);
  SetCurrentInput(FrontendInputFile());
}
