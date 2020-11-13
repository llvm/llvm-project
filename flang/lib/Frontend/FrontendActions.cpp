//===--- FrontendActions.cpp ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Frontend/FrontendActions.h"
#include "flang/Frontend/CompilerInstance.h"
#include "flang/Parser/parsing.h"
#include "flang/Parser/provenance.h"
#include "flang/Parser/source.h"

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
