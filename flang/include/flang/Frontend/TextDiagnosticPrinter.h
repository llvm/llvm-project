//===--- TextDiagnosticPrinter.h - Text Diagnostic Client -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is a concrete diagnostic client. In terminals that support it, the
// diagnostics are pretty-printed (colors + bold). The printing/flushing
// happens in HandleDiagnostics (usually called at the point when the
// diagnostic is generated).
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_FLANG_FRONTEND_TEXTDIAGNOSTICPRINTER_H
#define LLVM_FLANG_FRONTEND_TEXTDIAGNOSTICPRINTER_H

#include "clang/Basic/Diagnostic.h"
#include "llvm/ADT/IntrusiveRefCntPtr.h"
#include "llvm/Support/raw_ostream.h"

namespace clang {
class DiagnosticOptions;
class DiagnosticsEngine;
}; // namespace clang

using llvm::IntrusiveRefCntPtr;
using llvm::raw_ostream;

namespace Fortran::frontend {
class TextDiagnostic;

class TextDiagnosticPrinter : public clang::DiagnosticConsumer {
  raw_ostream &os_;
  llvm::IntrusiveRefCntPtr<clang::DiagnosticOptions> diagOpts_;

  /// A string to prefix to error messages.
  std::string prefix_;

public:
  TextDiagnosticPrinter(raw_ostream &os, clang::DiagnosticOptions *diags);
  ~TextDiagnosticPrinter() override;

  /// Set the diagnostic printer prefix string, which will be printed at the
  /// start of any diagnostics. If empty, no prefix string is used.
  void set_prefix(std::string value) { prefix_ = std::move(value); }

  void HandleDiagnostic(clang::DiagnosticsEngine::Level level,
      const clang::Diagnostic &info) override;
};

} // namespace Fortran::frontend

#endif
