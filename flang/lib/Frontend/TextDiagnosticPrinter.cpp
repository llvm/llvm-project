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
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

using namespace Fortran::frontend;

TextDiagnosticPrinter::TextDiagnosticPrinter(raw_ostream &diagOs,
                                             clang::DiagnosticOptions *diags)
    : os(diagOs), diagOpts(diags) {}

TextDiagnosticPrinter::~TextDiagnosticPrinter() {}

void TextDiagnosticPrinter::HandleDiagnostic(
    clang::DiagnosticsEngine::Level level, const clang::Diagnostic &info) {
  // Default implementation (Warnings/errors count).
  DiagnosticConsumer::HandleDiagnostic(level, info);

  // Render the diagnostic message into a temporary buffer eagerly. We'll use
  // this later as we print out the diagnostic to the terminal.
  llvm::SmallString<100> outStr;
  info.FormatDiagnostic(outStr);

  llvm::raw_svector_ostream diagMessageStream(outStr);

  if (!prefix.empty())
    os << prefix << ": ";

  // We only emit diagnostics in contexts that lack valid source locations.
  assert(!info.getLocation().isValid() &&
      "Diagnostics with valid source location are not supported");

  Fortran::frontend::TextDiagnostic::printDiagnosticLevel(os, level,
                                                          diagOpts->ShowColors);
  Fortran::frontend::TextDiagnostic::printDiagnosticMessage(
      os,
      /*IsSupplemental=*/level == clang::DiagnosticsEngine::Note,
      diagMessageStream.str(), diagOpts->ShowColors);

  os.flush();
  return;
}
