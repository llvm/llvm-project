//===--- FrontendAction.cpp -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Coding style: https://mlir.llvm.org/getting_started/DeveloperGuide/
//
//===----------------------------------------------------------------------===//

#include "flang/Frontend/FrontendAction.h"
#include "flang/Frontend/CompilerInstance.h"
#include "flang/Frontend/FrontendActions.h"
#include "flang/Frontend/FrontendOptions.h"
#include "flang/Frontend/FrontendPluginRegistry.h"
#include "clang/Basic/DiagnosticFrontend.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/VirtualFileSystem.h"

using namespace Fortran::frontend;

LLVM_INSTANTIATE_REGISTRY(FrontendPluginRegistry)

void FrontendAction::setCurrentInput(const FrontendInputFile &input) {
  this->currentInput = input;
}

// Call this method if BeginSourceFile fails.
// Deallocate compiler instance, input and output descriptors
static void beginSourceFileCleanUp(FrontendAction &fa, CompilerInstance &ci) {
  ci.clearOutputFiles(/*EraseFiles=*/true);
  fa.setCurrentInput(FrontendInputFile());
  fa.setInstance(nullptr);
}

bool FrontendAction::beginSourceFile(CompilerInstance &ci,
                                     const FrontendInputFile &realInput) {

  FrontendInputFile input(realInput);

  // Return immediately if the input file does not exist or is not a file. Note
  // that we cannot check this for input from stdin.
  if (input.getFile() != "-") {
    if (!llvm::sys::fs::is_regular_file(input.getFile())) {
      // Create an diagnostic ID to report
      unsigned diagID;
      if (llvm::vfs::getRealFileSystem()->exists(input.getFile())) {
        ci.getDiagnostics().Report(clang::diag::err_fe_error_reading)
            << input.getFile();
        diagID = ci.getDiagnostics().getCustomDiagID(
            clang::DiagnosticsEngine::Error, "%0 is not a regular file");
      } else {
        diagID = ci.getDiagnostics().getCustomDiagID(
            clang::DiagnosticsEngine::Error, "%0 does not exist");
      }

      // Report the diagnostic and return
      ci.getDiagnostics().Report(diagID) << input.getFile();
      beginSourceFileCleanUp(*this, ci);
      return false;
    }
  }

  assert(!instance && "Already processing a source file!");
  assert(!realInput.isEmpty() && "Unexpected empty filename!");
  setCurrentInput(realInput);
  setInstance(&ci);

  if (!ci.hasAllSources()) {
    beginSourceFileCleanUp(*this, ci);
    return false;
  }

  auto &invoc = ci.getInvocation();

  // Include command-line and predefined preprocessor macros. Use either:
  //  * `-cpp/-nocpp`, or
  //  * the file extension (if the user didn't express any preference)
  // to decide whether to include them or not.
  if ((invoc.getPreprocessorOpts().macrosFlag == PPMacrosFlag::Include) ||
      (invoc.getPreprocessorOpts().macrosFlag == PPMacrosFlag::Unknown &&
       getCurrentInput().getMustBePreprocessed())) {
    invoc.setDefaultPredefinitions();
    invoc.collectMacroDefinitions();
  }

  // Decide between fixed and free form (if the user didn't express any
  // preference, use the file extension to decide)
  if (invoc.getFrontendOpts().fortranForm == FortranForm::Unknown) {
    invoc.getFortranOpts().isFixedForm = getCurrentInput().getIsFixedForm();
  }

  if (!beginSourceFileAction()) {
    beginSourceFileCleanUp(*this, ci);
    return false;
  }

  return true;
}

bool FrontendAction::shouldEraseOutputFiles() {
  return getInstance().getDiagnostics().hasErrorOccurred();
}

llvm::Error FrontendAction::execute() {
  executeAction();

  return llvm::Error::success();
}

void FrontendAction::endSourceFile() {
  CompilerInstance &ci = getInstance();

  // Cleanup the output streams, and erase the output files if instructed by the
  // FrontendAction.
  ci.clearOutputFiles(/*EraseFiles=*/shouldEraseOutputFiles());

  setInstance(nullptr);
  setCurrentInput(FrontendInputFile());
}

bool FrontendAction::runPrescan() {
  CompilerInstance &ci = this->getInstance();
  std::string currentInputPath{getCurrentFileOrBufferName()};
  Fortran::parser::Options parserOptions = ci.getInvocation().getFortranOpts();

  if (ci.getInvocation().getFrontendOpts().fortranForm ==
      FortranForm::Unknown) {
    // Switch between fixed and free form format based on the input file
    // extension.
    //
    // Ideally we should have all Fortran options set before entering this
    // method (i.e. before processing any specific input files). However, we
    // can't decide between fixed and free form based on the file extension
    // earlier than this.
    parserOptions.isFixedForm = getCurrentInput().getIsFixedForm();
  }

  // Prescan. In case of failure, report and return.
  ci.getParsing().Prescan(currentInputPath, parserOptions);

  return !reportFatalScanningErrors();
}

bool FrontendAction::runParse() {
  CompilerInstance &ci = this->getInstance();

  // Parse. In case of failure, report and return.
  ci.getParsing().Parse(llvm::outs());

  if (reportFatalParsingErrors()) {
    return false;
  }

  // Report the diagnostics from getParsing
  ci.getParsing().messages().Emit(llvm::errs(), ci.getAllCookedSources());

  return true;
}

bool FrontendAction::runSemanticChecks() {
  CompilerInstance &ci = this->getInstance();
  std::optional<parser::Program> &parseTree{ci.getParsing().parseTree()};
  assert(parseTree && "Cannot run semantic checks without a parse tree!");

  // Prepare semantics
  ci.setSemantics(std::make_unique<Fortran::semantics::Semantics>(
      ci.getInvocation().getSemanticsContext(), *parseTree,
      ci.getInvocation().getDebugModuleDir()));
  auto &semantics = ci.getSemantics();

  // Run semantic checks
  semantics.Perform();

  if (reportFatalSemanticErrors()) {
    return false;
  }

  // Report the diagnostics from the semantic checks
  semantics.EmitMessages(ci.getSemaOutputStream());

  return true;
}

bool FrontendAction::generateRtTypeTables() {
  getInstance().setRtTyTables(
      std::make_unique<Fortran::semantics::RuntimeDerivedTypeTables>(
          BuildRuntimeDerivedTypeTables(
              getInstance().getInvocation().getSemanticsContext())));

  // The runtime derived type information table builder may find additional
  // semantic errors. Report them.
  if (reportFatalSemanticErrors()) {
    return false;
  }

  return true;
}

template <unsigned N>
bool FrontendAction::reportFatalErrors(const char (&message)[N]) {
  if (!instance->getParsing().messages().empty() &&
      (instance->getInvocation().getWarnAsErr() ||
       instance->getParsing().messages().AnyFatalError())) {
    const unsigned diagID = instance->getDiagnostics().getCustomDiagID(
        clang::DiagnosticsEngine::Error, message);
    instance->getDiagnostics().Report(diagID) << getCurrentFileOrBufferName();
    instance->getParsing().messages().Emit(llvm::errs(),
                                           instance->getAllCookedSources());
    return true;
  }
  return false;
}

bool FrontendAction::reportFatalSemanticErrors() {
  auto &diags = instance->getDiagnostics();
  auto &sema = instance->getSemantics();

  if (instance->getSemantics().AnyFatalError()) {
    unsigned diagID = diags.getCustomDiagID(clang::DiagnosticsEngine::Error,
                                            "Semantic errors in %0");
    diags.Report(diagID) << getCurrentFileOrBufferName();
    sema.EmitMessages(instance->getSemaOutputStream());

    return true;
  }

  return false;
}
