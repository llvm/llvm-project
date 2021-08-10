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
#include "flang/Frontend/PreprocessorOptions.h"
#include "flang/Lower/PFTBuilder.h"
#include "flang/Parser/dump-parse-tree.h"
#include "flang/Parser/parsing.h"
#include "flang/Parser/provenance.h"
#include "flang/Parser/source.h"
#include "flang/Parser/unparse.h"
#include "flang/Semantics/runtime-type-info.h"
#include "flang/Semantics/semantics.h"
#include "flang/Semantics/unparse-with-symbols.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ErrorHandling.h"
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

template <unsigned N>
static bool reportFatalErrors(
    const FrontendAction *act, const char (&message)[N]) {
  CompilerInstance &ci = act->instance();
  if (!ci.parsing().messages().empty() &&
      (ci.invocation().warnAsErr() ||
          ci.parsing().messages().AnyFatalError())) {
    const unsigned diagID = ci.diagnostics().getCustomDiagID(
        clang::DiagnosticsEngine::Error, message);
    ci.diagnostics().Report(diagID) << act->GetCurrentFileOrBufferName();
    ci.parsing().messages().Emit(llvm::errs(), ci.allCookedSources());
    return true;
  }
  return false;
}

inline bool reportFatalScanningErrors(const FrontendAction *act) {
  return reportFatalErrors(act, "Could not scan %0");
}

inline bool reportFatalParsingErrors(const FrontendAction *act) {
  return reportFatalErrors(act, "Could not parse %0");
}

bool PrescanAction::BeginSourceFileAction(CompilerInstance &c1) {
  CompilerInstance &ci = this->instance();
  std::string currentInputPath{GetCurrentFileOrBufferName()};
  Fortran::parser::Options parserOptions = ci.invocation().fortranOpts();

  // Prescan. In case of failure, report and return.
  ci.parsing().Prescan(currentInputPath, parserOptions);

  return !reportFatalScanningErrors(this);
}

bool PrescanAndParseAction::BeginSourceFileAction(CompilerInstance &c1) {
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

  if (reportFatalScanningErrors(this))
    return false;

  // Parse. In case of failure, report and return.
  ci.parsing().Parse(llvm::outs());

  if (reportFatalParsingErrors(this))
    return false;

  // Report the diagnostics from parsing
  ci.parsing().messages().Emit(llvm::errs(), ci.allCookedSources());

  return true;
}

bool PrescanAndSemaAction::BeginSourceFileAction(CompilerInstance &c1) {
  CompilerInstance &ci = this->instance();
  std::string currentInputPath{GetCurrentFileOrBufferName()};
  Fortran::parser::Options parserOptions = ci.invocation().fortranOpts();

  // Prescan. In case of failure, report and return.
  ci.parsing().Prescan(currentInputPath, parserOptions);

  if (reportFatalScanningErrors(this))
    return false;

  // Parse. In case of failure, report and return.
  ci.parsing().Parse(llvm::outs());

  if (reportFatalParsingErrors(this))
    return false;

  // Report the diagnostics from parsing
  ci.parsing().messages().Emit(llvm::errs(), ci.allCookedSources());

  auto &parseTree{*ci.parsing().parseTree()};

  // Prepare semantics
  setSemantics(std::make_unique<Fortran::semantics::Semantics>(
      ci.invocation().semanticsContext(), parseTree,
      ci.invocation().debugModuleDir()));
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

  // Format or dump the prescanner's output
  CompilerInstance &ci = this->instance();
  if (ci.invocation().preprocessorOpts().noReformat) {
    ci.parsing().DumpCookedChars(outForPP);
  } else {
    ci.parsing().EmitPreprocessedSource(
        outForPP, !ci.invocation().preprocessorOpts().noLineDirectives);
  }

  // If a pre-defined output stream exists, dump the preprocessed content there
  if (!ci.IsOutputStreamNull()) {
    // Send the output to the pre-defined output buffer.
    ci.WriteOutputStream(outForPP.str());
    return;
  }

  // Print diagnostics from the prescanner
  ci.parsing().messages().Emit(llvm::errs(), ci.allCookedSources());

  // Create a file and save the preprocessed output there
  if (auto os{ci.CreateDefaultOutputFile(
          /*Binary=*/true, /*InFile=*/GetCurrentFileOrBufferName())}) {
    (*os) << outForPP.str();
  } else {
    llvm::errs() << "Unable to create the output file\n";
  }
}

void DebugDumpProvenanceAction::ExecuteAction() {
  this->instance().parsing().DumpProvenance(llvm::outs());
}

void ParseSyntaxOnlyAction::ExecuteAction() {
  reportFatalSemanticErrors(semantics(), this->instance().diagnostics(),
      GetCurrentFileOrBufferName());
}

void DebugUnparseNoSemaAction::ExecuteAction() {
  auto &invoc = this->instance().invocation();
  auto &parseTree{instance().parsing().parseTree()};

  // TODO: Options should come from CompilerInvocation
  Unparse(llvm::outs(), *parseTree,
      /*encoding=*/Fortran::parser::Encoding::UTF_8,
      /*capitalizeKeywords=*/true, /*backslashEscapes=*/false,
      /*preStatement=*/nullptr,
      invoc.useAnalyzedObjectsForUnparse() ? &invoc.asFortran() : nullptr);
}

void DebugUnparseAction::ExecuteAction() {
  auto &invoc = this->instance().invocation();
  auto &parseTree{instance().parsing().parseTree()};

  CompilerInstance &ci = this->instance();
  auto os{ci.CreateDefaultOutputFile(
      /*Binary=*/false, /*InFile=*/GetCurrentFileOrBufferName())};

  // TODO: Options should come from CompilerInvocation
  Unparse(*os, *parseTree,
      /*encoding=*/Fortran::parser::Encoding::UTF_8,
      /*capitalizeKeywords=*/true, /*backslashEscapes=*/false,
      /*preStatement=*/nullptr,
      invoc.useAnalyzedObjectsForUnparse() ? &invoc.asFortran() : nullptr);

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
  CompilerInstance &ci = this->instance();
  auto &semantics = this->semantics();

  auto tables{Fortran::semantics::BuildRuntimeDerivedTypeTables(
      instance().invocation().semanticsContext())};
  // The runtime derived type information table builder may find and report
  // semantic errors. So it is important that we report them _after_
  // BuildRuntimeDerivedTypeTables is run.
  reportFatalSemanticErrors(
      semantics, this->instance().diagnostics(), GetCurrentFileOrBufferName());

  if (!tables.schemata) {
    unsigned DiagID =
        ci.diagnostics().getCustomDiagID(clang::DiagnosticsEngine::Error,
            "could not find module file for __fortran_type_info");
    ci.diagnostics().Report(DiagID);
    llvm::errs() << "\n";
  }

  // Dump symbols
  semantics.DumpSymbols(llvm::outs());
}

void DebugDumpAllAction::ExecuteAction() {
  CompilerInstance &ci = this->instance();

  // Dump parse tree
  auto &parseTree{instance().parsing().parseTree()};
  llvm::outs() << "========================";
  llvm::outs() << " Flang: parse tree dump ";
  llvm::outs() << "========================\n";
  Fortran::parser::DumpTree(
      llvm::outs(), parseTree, &ci.invocation().asFortran());

  auto &semantics = this->semantics();
  auto tables{Fortran::semantics::BuildRuntimeDerivedTypeTables(
      instance().invocation().semanticsContext())};
  // The runtime derived type information table builder may find and report
  // semantic errors. So it is important that we report them _after_
  // BuildRuntimeDerivedTypeTables is run.
  reportFatalSemanticErrors(
      semantics, this->instance().diagnostics(), GetCurrentFileOrBufferName());

  if (!tables.schemata) {
    unsigned DiagID =
        ci.diagnostics().getCustomDiagID(clang::DiagnosticsEngine::Error,
            "could not find module file for __fortran_type_info");
    ci.diagnostics().Report(DiagID);
    llvm::errs() << "\n";
  }

  // Dump symbols
  llvm::outs() << "=====================";
  llvm::outs() << " Flang: symbols dump ";
  llvm::outs() << "=====================\n";
  semantics.DumpSymbols(llvm::outs());
}

void DebugDumpParseTreeNoSemaAction::ExecuteAction() {
  auto &parseTree{instance().parsing().parseTree()};

  // Dump parse tree
  Fortran::parser::DumpTree(
      llvm::outs(), parseTree, &this->instance().invocation().asFortran());
}

void DebugDumpParseTreeAction::ExecuteAction() {
  auto &parseTree{instance().parsing().parseTree()};

  // Dump parse tree
  Fortran::parser::DumpTree(
      llvm::outs(), parseTree, &this->instance().invocation().asFortran());

  // Report fatal semantic errors
  reportFatalSemanticErrors(semantics(), this->instance().diagnostics(),
      GetCurrentFileOrBufferName());
}

void DebugMeasureParseTreeAction::ExecuteAction() {
  CompilerInstance &ci = this->instance();

  // Parse. In case of failure, report and return.
  ci.parsing().Parse(llvm::outs());

  if (!ci.parsing().messages().empty() &&
      (ci.invocation().warnAsErr() ||
          ci.parsing().messages().AnyFatalError())) {
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

void GetDefinitionAction::ExecuteAction() {
  // Report and exit if fatal semantic errors are present
  if (reportFatalSemanticErrors(semantics(), this->instance().diagnostics(),
          GetCurrentFileOrBufferName()))
    return;

  CompilerInstance &ci = this->instance();
  parser::AllCookedSources &cs = ci.allCookedSources();
  unsigned diagID = ci.diagnostics().getCustomDiagID(
      clang::DiagnosticsEngine::Error, "Symbol not found");

  auto gdv = ci.invocation().frontendOpts().getDefVals;
  auto charBlock{cs.GetCharBlockFromLineAndColumns(
      gdv.line, gdv.startColumn, gdv.endColumn)};
  if (!charBlock) {
    ci.diagnostics().Report(diagID);
    return;
  }

  llvm::outs() << "String range: >" << charBlock->ToString() << "<\n";

  auto *symbol{ci.invocation()
                   .semanticsContext()
                   .FindScope(*charBlock)
                   .FindSymbol(*charBlock)};
  if (!symbol) {
    ci.diagnostics().Report(diagID);
    return;
  }

  llvm::outs() << "Found symbol name: " << symbol->name().ToString() << "\n";

  auto sourceInfo{cs.GetSourcePositionRange(symbol->name())};
  if (!sourceInfo) {
    llvm_unreachable(
        "Failed to obtain SourcePosition."
        "TODO: Please, write a test and replace this with a diagnostic!");
    return;
  }

  llvm::outs() << "Found symbol name: " << symbol->name().ToString() << "\n";
  llvm::outs() << symbol->name().ToString() << ": "
               << sourceInfo->first.file.path() << ", "
               << sourceInfo->first.line << ", " << sourceInfo->first.column
               << "-" << sourceInfo->second.column << "\n";
}

void GetSymbolsSourcesAction::ExecuteAction() {
  // Report and exit if fatal semantic errors are present
  if (reportFatalSemanticErrors(semantics(), this->instance().diagnostics(),
          GetCurrentFileOrBufferName()))
    return;

  semantics().DumpSymbolsSources(llvm::outs());
}

void EmitObjAction::ExecuteAction() {
  CompilerInstance &ci = this->instance();
  unsigned DiagID = ci.diagnostics().getCustomDiagID(
      clang::DiagnosticsEngine::Error, "code-generation is not available yet");
  ci.diagnostics().Report(DiagID);
}

void InitOnlyAction::ExecuteAction() {
  CompilerInstance &ci = this->instance();
  unsigned DiagID =
      ci.diagnostics().getCustomDiagID(clang::DiagnosticsEngine::Warning,
          "Use `-init-only` for testing purposes only");
  ci.diagnostics().Report(DiagID);
}
