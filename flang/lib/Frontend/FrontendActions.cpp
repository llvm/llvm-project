//===--- FrontendActions.cpp ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Frontend/FrontendActions.h"
#include "flang/Common/default-kinds.h"
#include "flang/Frontend/CompilerInstance.h"
#include "flang/Frontend/FrontendOptions.h"
#include "flang/Lower/PFTBuilder.h"
#include "flang/Parser/dump-parse-tree.h"
#include "flang/Parser/parsing.h"
#include "flang/Parser/provenance.h"
#include "flang/Parser/source.h"
#include "flang/Parser/unparse.h"
#include "flang/Semantics/semantics.h"
#include "flang/Semantics/unparse-with-symbols.h"
#include "llvm/ADT/StringRef.h"
#include <clang/Basic/Diagnostic.h>
#include <memory>

using namespace Fortran::frontend;

/// Report fatal semantic errors if present.
///
/// \param semantics The semantics instance
/// \param diags The diagnostics engine instance
/// \param bufferName The file or buffer name
///
/// \return True if fatal semantic errors are present, false if not
bool reportFatalSemanticErrors(const Fortran::semantics::Semantics &semantics,
    clang::DiagnosticsEngine &diags, const llvm::StringRef &bufferName) {
  if (semantics.AnyFatalError()) {
    unsigned DiagID = diags.getCustomDiagID(
        clang::DiagnosticsEngine::Error, "Semantic errors in %0");
    diags.Report(DiagID) << bufferName;
    return true;
  }
  return false;
}

bool PrescanAction::BeginSourceFileAction(CompilerInstance &c1) {
  CompilerInstance &ci = this->instance();

  std::string currentInputPath{GetCurrentFileOrBufferName()};

  Fortran::parser::Options parserOptions = ci.invocation().fortranOpts();

  if (ci.invocation().frontendOpts().fortranForm_ == FortranForm::Unknown) {
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

  if (ci.parsing().messages().AnyFatalError()) {
    const unsigned diagID = ci.diagnostics().getCustomDiagID(
        clang::DiagnosticsEngine::Error, "Could not scan %0");
    ci.diagnostics().Report(diagID) << GetCurrentFileOrBufferName();
    ci.parsing().messages().Emit(llvm::errs(), ci.allCookedSources());

    return false;
  }

  return true;
}

bool PrescanAndSemaAction::BeginSourceFileAction(CompilerInstance &c1) {
  CompilerInstance &ci = this->instance();

  std::string currentInputPath{GetCurrentFileOrBufferName()};

  Fortran::parser::Options parserOptions = ci.invocation().fortranOpts();

  if (ci.invocation().frontendOpts().fortranForm_ == FortranForm::Unknown) {
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

  if (ci.parsing().messages().AnyFatalError()) {
    const unsigned diagID = ci.diagnostics().getCustomDiagID(
        clang::DiagnosticsEngine::Error, "Could not scan %0");
    ci.diagnostics().Report(diagID) << GetCurrentFileOrBufferName();
    ci.parsing().messages().Emit(llvm::errs(), ci.allCookedSources());

    return false;
  }

  // Parse. In case of failure, report and return.
  ci.parsing().Parse(llvm::outs());

  if (ci.parsing().messages().AnyFatalError()) {
    unsigned diagID = ci.diagnostics().getCustomDiagID(
        clang::DiagnosticsEngine::Error, "Could not parse %0");
    ci.diagnostics().Report(diagID) << GetCurrentFileOrBufferName();

    ci.parsing().messages().Emit(
        llvm::errs(), this->instance().allCookedSources());
    return false;
  }

  // Report the diagnostics from parsing
  ci.parsing().messages().Emit(llvm::errs(), ci.allCookedSources());

  auto &parseTree{*ci.parsing().parseTree()};

  // Prepare semantics
  setSemantics(std::make_unique<Fortran::semantics::Semantics>(
      ci.invocation().semanticsContext(), parseTree,
      ci.parsing().cooked().AsCharBlock(), ci.invocation().debugModuleDir()));
  auto &semantics = this->semantics();

  // Run semantic checks
  semantics.Perform();

  // Report the diagnostics from the semantic checks
  semantics.EmitMessages(ci.semaOutputStream());
  return true;
}

void InputOutputTestAction::ExecuteAction() {
  CompilerInstance &ci = instance();

  // Create a stream for errors
  std::string buf;
  llvm::raw_string_ostream error_stream{buf};

  // Read the input file
  Fortran::parser::AllSources &allSources{ci.allSources()};
  std::string path{GetCurrentFileOrBufferName()};
  const Fortran::parser::SourceFile *sf;
  if (path == "-")
    sf = allSources.ReadStandardInput(error_stream);
  else
    sf = allSources.Open(path, error_stream, std::optional<std::string>{"."s});
  llvm::ArrayRef<char> fileContent = sf->content();

  // Output file descriptor to receive the contents of the input file.
  std::unique_ptr<llvm::raw_ostream> os;

  // Copy the contents from the input file to the output file
  if (!ci.IsOutputStreamNull()) {
    // An output stream (outputStream_) was set earlier
    ci.WriteOutputStream(fileContent.data());
  } else {
    // No pre-set output stream - create an output file
    os = ci.CreateDefaultOutputFile(
        /*binary=*/true, GetCurrentFileOrBufferName(), "txt");
    if (!os)
      return;
    (*os) << fileContent.data();
  }
}

void PrintPreprocessedAction::ExecuteAction() {
  std::string buf;
  llvm::raw_string_ostream outForPP{buf};

  // Run the preprocessor
  CompilerInstance &ci = this->instance();
  ci.parsing().DumpCookedChars(outForPP);

  // If a pre-defined output stream exists, dump the preprocessed content there
  if (!ci.IsOutputStreamNull()) {
    // Send the output to the pre-defined output buffer.
    ci.WriteOutputStream(outForPP.str());
    return;
  }

  // Print diagnostics from the preprocessor
  ci.parsing().messages().Emit(llvm::errs(), ci.allCookedSources());

  // Create a file and save the preprocessed output there
  if (auto os{ci.CreateDefaultOutputFile(
          /*Binary=*/true, /*InFile=*/GetCurrentFileOrBufferName())}) {
    (*os) << outForPP.str();
  } else {
    llvm::errs() << "Unable to create the output file\n";
    return;
  }
}

void DebugDumpProvenanceAction::ExecuteAction() {
  this->instance().parsing().DumpProvenance(llvm::outs());
}

void ParseSyntaxOnlyAction::ExecuteAction() {
  reportFatalSemanticErrors(semantics(), this->instance().diagnostics(),
      GetCurrentFileOrBufferName());
}

void DebugUnparseAction::ExecuteAction() {
  auto &parseTree{instance().parsing().parseTree()};
  Fortran::parser::AnalyzedObjectsAsFortran asFortran =
      Fortran::frontend::getBasicAsFortran();

  // TODO: Options should come from CompilerInvocation
  Unparse(llvm::outs(), *parseTree,
      /*encoding=*/Fortran::parser::Encoding::UTF_8,
      /*capitalizeKeywords=*/true, /*backslashEscapes=*/false,
      /*preStatement=*/nullptr, &asFortran);

  // Report fatal semantic errors
  reportFatalSemanticErrors(semantics(), this->instance().diagnostics(),
      GetCurrentFileOrBufferName());
}

void DebugUnparseWithSymbolsAction::ExecuteAction() {
  auto &parseTree{*instance().parsing().parseTree()};

  Fortran::semantics::UnparseWithSymbols(
      llvm::outs(), parseTree, /*encoding=*/Fortran::parser::Encoding::UTF_8);

  // Report fatal semantic errors
  reportFatalSemanticErrors(semantics(), this->instance().diagnostics(),
      GetCurrentFileOrBufferName());
}

void DebugDumpSymbolsAction::ExecuteAction() {
  auto &semantics = this->semantics();

  // Dump symbols
  semantics.DumpSymbols(llvm::outs());
  // Report fatal semantic errors
  reportFatalSemanticErrors(
      semantics, this->instance().diagnostics(), GetCurrentFileOrBufferName());
}

void DebugDumpParseTreeAction::ExecuteAction() {
  auto &parseTree{instance().parsing().parseTree()};
  Fortran::parser::AnalyzedObjectsAsFortran asFortran =
      Fortran::frontend::getBasicAsFortran();

  // Dump parse tree
  Fortran::parser::DumpTree(llvm::outs(), parseTree, &asFortran);
  // Report fatal semantic errors
  reportFatalSemanticErrors(semantics(), this->instance().diagnostics(),
      GetCurrentFileOrBufferName());
}

void DebugMeasureParseTreeAction::ExecuteAction() {
  CompilerInstance &ci = this->instance();

  // Parse. In case of failure, report and return.
  ci.parsing().Parse(llvm::outs());

  if (ci.parsing().messages().AnyFatalError()) {
    unsigned diagID = ci.diagnostics().getCustomDiagID(
        clang::DiagnosticsEngine::Error, "Could not parse %0");
    ci.diagnostics().Report(diagID) << GetCurrentFileOrBufferName();

    ci.parsing().messages().Emit(
        llvm::errs(), this->instance().allCookedSources());
    return;
  }

  // Report the diagnostics from parsing
  ci.parsing().messages().Emit(llvm::errs(), ci.allCookedSources());

  auto &parseTree{*ci.parsing().parseTree()};

  // Measure the parse tree
  MeasurementVisitor visitor;
  Fortran::parser::Walk(parseTree, visitor);
  llvm::outs() << "Parse tree comprises " << visitor.objects
               << " objects and occupies " << visitor.bytes
               << " total bytes.\n";
}

void DebugPreFIRTreeAction::ExecuteAction() {
  CompilerInstance &ci = this->instance();
  // Report and exit if fatal semantic errors are present
  if (reportFatalSemanticErrors(
          semantics(), ci.diagnostics(), GetCurrentFileOrBufferName())) {
    return;
  }

  auto &parseTree{*ci.parsing().parseTree()};

  // Dump pre-FIR tree
  if (auto ast{Fortran::lower::createPFT(
          parseTree, ci.invocation().semanticsContext())}) {
    Fortran::lower::dumpPFT(llvm::outs(), *ast);
  } else {
    unsigned diagID = ci.diagnostics().getCustomDiagID(
        clang::DiagnosticsEngine::Error, "Pre FIR Tree is NULL.");
    ci.diagnostics().Report(diagID);
  }
}

void DebugDumpParsingLogAction::ExecuteAction() {
  CompilerInstance &ci = this->instance();

  ci.parsing().Parse(llvm::errs());
  ci.parsing().DumpParsingLog(llvm::outs());
}

void EmitObjAction::ExecuteAction() {
  CompilerInstance &ci = this->instance();
  unsigned DiagID = ci.diagnostics().getCustomDiagID(
      clang::DiagnosticsEngine::Error, "code-generation is not available yet");
  ci.diagnostics().Report(DiagID);
}
