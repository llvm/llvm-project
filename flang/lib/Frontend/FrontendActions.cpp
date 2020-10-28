//===--- FrontendActions.cpp ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "flang/Frontend/FrontendActions.h"
#include "flang/Common/Fortran-features.h"
#include "flang/Common/default-kinds.h"
#include "flang/Frontend/CompilerInstance.h"
#include "flang/Parser/source.h"
#include "clang/Serialization/PCHContainerOperations.h"

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
