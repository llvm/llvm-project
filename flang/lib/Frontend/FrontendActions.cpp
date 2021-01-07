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
#include "flang/Parser/parsing.h"
#include "flang/Parser/provenance.h"
#include "flang/Parser/source.h"
#include "flang/Semantics/semantics.h"

using namespace Fortran::frontend;

void InputOutputTestAction::ExecuteAction() {

  // Get the name of the file from FrontendInputFile current.
  std::string path{GetCurrentFileOrBufferName()};
  std::string buf;
  llvm::raw_string_ostream error_stream{buf};
  bool binaryMode = true;

  // Set/store input file info into CompilerInstance.
  CompilerInstance &ci = instance();
  Fortran::parser::AllSources &allSources{ci.allSources()};
  const Fortran::parser::SourceFile *sf;
  sf = allSources.Open(path, error_stream);
  llvm::ArrayRef<char> fileContent = sf->content();

  // Output file descriptor to receive the content of input file.
  std::unique_ptr<llvm::raw_ostream> os;

  // Do not write on the output file if using outputStream_.
  if (ci.IsOutputStreamNull()) {
    os = ci.CreateDefaultOutputFile(
        binaryMode, GetCurrentFileOrBufferName(), "txt");
    if (!os)
      return;
    (*os) << fileContent.data();
  } else {
    ci.WriteOutputStream(fileContent.data());
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

  // Create a file and save the preprocessed output there
  if (auto os{ci.CreateDefaultOutputFile(
          /*Binary=*/true, /*InFile=*/GetCurrentFileOrBufferName())}) {
    (*os) << outForPP.str();
  } else {
    llvm::errs() << "Unable to create the output file\n";
    return;
  }
}

void ParseSyntaxOnlyAction::ExecuteAction() {
  CompilerInstance &ci = this->instance();

  // TODO: These should be specifiable by users. For now just use the defaults.
  common::LanguageFeatureControl features;
  Fortran::common::IntrinsicTypeDefaultKinds defaultKinds;

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

  auto &parseTree{*ci.parsing().parseTree()};

  // Prepare semantics
  Fortran::semantics::SemanticsContext semanticsContext{
      defaultKinds, features, ci.allCookedSources()};
  Fortran::semantics::Semantics semantics{
      semanticsContext, parseTree, ci.parsing().cooked().AsCharBlock()};

  // Run semantic checks
  semantics.Perform();

  // Report the diagnostics from the semantic checks
  semantics.EmitMessages(ci.semaOutputStream());

  if (semantics.AnyFatalError()) {
    unsigned DiagID = ci.diagnostics().getCustomDiagID(
        clang::DiagnosticsEngine::Error, "Semantic errors in %0");
    ci.diagnostics().Report(DiagID) << GetCurrentFileOrBufferName();
  }
}

void EmitObjAction::ExecuteAction() {
  CompilerInstance &ci = this->instance();
  unsigned DiagID = ci.diagnostics().getCustomDiagID(
      clang::DiagnosticsEngine::Error, "code-generation is not available yet");
  ci.diagnostics().Report(DiagID);
}
