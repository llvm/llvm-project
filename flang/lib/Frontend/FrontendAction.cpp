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
#include "flang/Frontend/FrontendPluginRegistry.h"
#include "flang/FrontendTool/Utils.h"
#include "clang/Basic/DiagnosticFrontend.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/VirtualFileSystem.h"

using namespace Fortran::frontend;

LLVM_INSTANTIATE_REGISTRY(FrontendPluginRegistry)

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

  auto &invoc = ci.invocation();

  // Include command-line and predefined preprocessor macros. Use either:
  //  * `-cpp/-nocpp`, or
  //  * the file extension (if the user didn't express any preference)
  // to decide whether to include them or not.
  if ((invoc.preprocessorOpts().macrosFlag == PPMacrosFlag::Include) ||
      (invoc.preprocessorOpts().macrosFlag == PPMacrosFlag::Unknown &&
          currentInput().MustBePreprocessed())) {
    invoc.SetDefaultPredefinitions();
    invoc.CollectMacroDefinitions();
  }

  // Decide between fixed and free form (if the user didn't express any
  // preference, use the file extension to decide)
  if (invoc.frontendOpts().fortranForm == FortranForm::Unknown) {
    invoc.fortranOpts().isFixedForm = currentInput().IsFixedForm();
  }

  if (!BeginSourceFileAction()) {
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

bool FrontendAction::RunPrescan() {
  CompilerInstance &ci = this->instance();
  std::string currentInputPath{GetCurrentFileOrBufferName()};
  Fortran::parser::Options parserOptions = ci.invocation().fortranOpts();

  if (ci.invocation().frontendOpts().fortranForm == FortranForm::Unknown) {
    // Switch between fixed and free form format based on the input file
    // extension.
    //
    // Ideally we should have all Fortran options set before entering this
    // method (i.e. before processing any specific input files). However, we
    // can't decide between fixed and free form based on the file extension
    // earlier than this.
    parserOptions.isFixedForm = currentInput().IsFixedForm();
  }

  // Prescan. In case of failure, report and return.
  ci.parsing().Prescan(currentInputPath, parserOptions);

  return !reportFatalScanningErrors();
}

bool FrontendAction::RunParse() {
  CompilerInstance &ci = this->instance();

  // Parse. In case of failure, report and return.
  ci.parsing().Parse(llvm::outs());

  if (reportFatalParsingErrors()) {
    return false;
  }

  // Report the diagnostics from parsing
  ci.parsing().messages().Emit(llvm::errs(), ci.allCookedSources());

  return true;
}

bool FrontendAction::RunSemanticChecks() {
  CompilerInstance &ci = this->instance();
  std::optional<parser::Program> &parseTree{ci.parsing().parseTree()};
  assert(parseTree && "Cannot run semantic checks without a parse tree!");

  // Prepare semantics
  ci.SetSemantics(std::make_unique<Fortran::semantics::Semantics>(
      ci.invocation().semanticsContext(), *parseTree,
      ci.invocation().debugModuleDir()));
  auto &semantics = ci.semantics();

  // Run semantic checks
  semantics.Perform();

  if (reportFatalSemanticErrors()) {
    return false;
  }

  // Report the diagnostics from the semantic checks
  semantics.EmitMessages(ci.semaOutputStream());

  return true;
}

template <unsigned N>
bool FrontendAction::reportFatalErrors(const char (&message)[N]) {
  if (!instance_->parsing().messages().empty() &&
      (instance_->invocation().warnAsErr() ||
          instance_->parsing().messages().AnyFatalError())) {
    const unsigned diagID = instance_->diagnostics().getCustomDiagID(
        clang::DiagnosticsEngine::Error, message);
    instance_->diagnostics().Report(diagID) << GetCurrentFileOrBufferName();
    instance_->parsing().messages().Emit(
        llvm::errs(), instance_->allCookedSources());
    return true;
  }
  return false;
}

bool FrontendAction::reportFatalSemanticErrors() {
  auto &diags = instance_->diagnostics();
  auto &sema = instance_->semantics();

  if (instance_->semantics().AnyFatalError()) {
    unsigned DiagID = diags.getCustomDiagID(
        clang::DiagnosticsEngine::Error, "Semantic errors in %0");
    diags.Report(DiagID) << GetCurrentFileOrBufferName();
    sema.EmitMessages(instance_->semaOutputStream());

    return true;
  }

  return false;
}
