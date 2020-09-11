//===- CompilerInvocation.h - Compiler Invocation Helper Data ---*- C -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_FLANG_FRONTEND_COMPILERINVOCATION_H
#define LLVM_FLANG_FRONTEND_COMPILERINVOCATION_H

#include "flang/Frontend/FrontendOptions.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/DiagnosticOptions.h"

namespace Fortran::frontend {
class CompilerInvocationBase {
public:
  /// Options controlling the diagnostic engine.$
  llvm::IntrusiveRefCntPtr<clang::DiagnosticOptions> diagnosticOpts_;

  CompilerInvocationBase();
  CompilerInvocationBase(const CompilerInvocationBase &x);
  ~CompilerInvocationBase();

  clang::DiagnosticOptions &GetDiagnosticOpts() {
    return *diagnosticOpts_.get();
  }
  const clang::DiagnosticOptions &GetDiagnosticOpts() const {
    return *diagnosticOpts_.get();
  }
};

class CompilerInvocation : public CompilerInvocationBase {
  /// Options controlling the frontend itself.
  FrontendOptions frontendOpts_;

public:
  CompilerInvocation() = default;

  FrontendOptions &GetFrontendOpts() { return frontendOpts_; }
  const FrontendOptions &GetFrontendOpts() const { return frontendOpts_; }

  /// Create a compiler invocation from a list of input options.
  /// \returns true on success.
  /// \returns false if an error was encountered while parsing the arguments
  /// \param [out] res - The resulting invocation.
  static bool CreateFromArgs(CompilerInvocation &res,
      llvm::ArrayRef<const char *> commandLineArgs,
      clang::DiagnosticsEngine &diags);
};

} // end namespace Fortran::frontend
#endif // LLVM_FLANG_FRONTEND_COMPILERINVOCATION_H
