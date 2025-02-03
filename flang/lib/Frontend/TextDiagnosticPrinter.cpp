//===--- TextDiagnosticPrinter.cpp - Diagnostic Printer -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This diagnostic client prints out their diagnostic messages.
//
//===----------------------------------------------------------------------===//
//
// Coding style: https://mlir.llvm.org/getting_started/DeveloperGuide/
//
//===----------------------------------------------------------------------===//

#include "flang/Frontend/TextDiagnosticPrinter.h"
#include "flang/Frontend/TextDiagnostic.h"
#include "clang/Basic/DiagnosticOptions.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

using namespace Fortran::frontend;

TextDiagnosticPrinter::TextDiagnosticPrinter(raw_ostream &diagOs,
                                             clang::DiagnosticOptions *diags)
    : os(diagOs), diagOpts(diags) {}

TextDiagnosticPrinter::~TextDiagnosticPrinter() {}

// For remarks only, print the remark option and pass name that was used to a
// raw_ostream. This also supports warnings from invalid remark arguments
// provided.
static void printRemarkOption(llvm::raw_ostream &os,
                              clang::DiagnosticsEngine::Level level,
                              const clang::Diagnostic &info) {
  llvm::StringRef opt =
      info.getDiags()->getDiagnosticIDs()->getWarningOptionForDiag(
          info.getID());
  if (!opt.empty()) {
    // We still need to check if the level is a Remark since, an unknown option
    // warning could be printed i.e. [-Wunknown-warning-option]
    os << " [" << (level == clang::DiagnosticsEngine::Remark ? "-R" : "-W")
       << opt;
    llvm::StringRef optValue = info.getFlagValue();
    if (!optValue.empty())
      os << "=" << optValue;
    os << ']';
  }
}

// For remarks only, if we are receiving a message of this format
// [file location with line and column];;[path to file];;[the remark message]
// then print the absolute file path, line and column number.
void TextDiagnosticPrinter::printLocForRemarks(
    llvm::raw_svector_ostream &diagMessageStream, llvm::StringRef &diagMsg) {
  // split incoming string to get the absolute path and filename in the
  // case we are receiving optimization remarks from BackendRemarkConsumer
  diagMsg = diagMessageStream.str();
  llvm::StringRef delimiter = ";;";

  size_t pos = 0;
  llvm::SmallVector<llvm::StringRef> tokens;
  while ((pos = diagMsg.find(delimiter)) != std::string::npos) {
    tokens.push_back(diagMsg.substr(0, pos));
    diagMsg = diagMsg.drop_front(pos + delimiter.size());
  }

  // tokens will always be of size 2 in the case of optimization
  // remark message received
  if (tokens.size() == 2) {
    // Extract absolute path
    llvm::SmallString<128> absPath = llvm::sys::path::relative_path(tokens[1]);
    llvm::sys::path::remove_filename(absPath);
    // Add the last separator before the file name
    llvm::sys::path::append(absPath, llvm::sys::path::get_separator());
    llvm::sys::path::make_preferred(absPath);

    // Used for changing only the bold attribute
    if (diagOpts->ShowColors)
      os.changeColor(llvm::raw_ostream::SAVEDCOLOR, true);

    // Print path, file name, line and column
    os << absPath << tokens[0] << ": ";
  }
}

void TextDiagnosticPrinter::HandleDiagnostic(
    clang::DiagnosticsEngine::Level level, const clang::Diagnostic &info) {
  // Default implementation (Warnings/errors count).
  DiagnosticConsumer::HandleDiagnostic(level, info);

  // Render the diagnostic message into a temporary buffer eagerly. We'll use
  // this later as we print out the diagnostic to the terminal.
  llvm::SmallString<100> outStr;
  info.FormatDiagnostic(outStr);

  llvm::raw_svector_ostream diagMessageStream(outStr);
  printRemarkOption(diagMessageStream, level, info);

  if (!prefix.empty())
    os << prefix << ": ";

  // We only emit diagnostics in contexts that lack valid source locations.
  assert(!info.getLocation().isValid() &&
         "Diagnostics with valid source location are not supported");

  llvm::StringRef diagMsg;
  printLocForRemarks(diagMessageStream, diagMsg);

  Fortran::frontend::TextDiagnostic::printDiagnosticLevel(os, level,
                                                          diagOpts->ShowColors);
  Fortran::frontend::TextDiagnostic::printDiagnosticMessage(
      os,
      /*IsSupplemental=*/level == clang::DiagnosticsEngine::Note, diagMsg,
      diagOpts->ShowColors);

  os.flush();
}
