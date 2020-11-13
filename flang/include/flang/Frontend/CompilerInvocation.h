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
#include "flang/Parser/parsing.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/DiagnosticOptions.h"
#include "llvm/Option/ArgList.h"

namespace Fortran::frontend {

/// Fill out Opts based on the options given in Args.
///
/// When errors are encountered, return false and, if Diags is non-null,
/// report the error(s).
bool ParseDiagnosticArgs(clang::DiagnosticOptions &opts,
    llvm::opt::ArgList &args, bool defaultDiagColor = true);

class CompilerInvocationBase {
public:
  /// Options controlling the diagnostic engine.
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
  /// Options for the frontend driver
  // TODO: Merge with or translate to parserOpts_. We shouldn't need two sets of
  // options.
  FrontendOptions frontendOpts_;

  /// Options for Flang parser
  // TODO: Merge with or translate to frontendOpts_. We shouldn't need two sets
  // of options.
  Fortran::parser::Options parserOpts_;

public:
  CompilerInvocation() = default;

  FrontendOptions &frontendOpts() { return frontendOpts_; }
  const FrontendOptions &frontendOpts() const { return frontendOpts_; }

  Fortran::parser::Options &fortranOpts() { return parserOpts_; }
  const Fortran::parser::Options &fortranOpts() const { return parserOpts_; }

  /// Create a compiler invocation from a list of input options.
  /// \returns true on success.
  /// \returns false if an error was encountered while parsing the arguments
  /// \param [out] res - The resulting invocation.
  static bool CreateFromArgs(CompilerInvocation &res,
      llvm::ArrayRef<const char *> commandLineArgs,
      clang::DiagnosticsEngine &diags);

  /// Set the Fortran options to predifined defaults. These defaults are
  /// consistend with f18/f18.cpp.
  // TODO: We should map frontendOpts_ to parserOpts_ instead. For that, we
  // need to extend frontendOpts_ first. Next, we need to add the corresponding
  // compiler driver options in libclangDriver.
  void SetDefaultFortranOpts();
};

} // end namespace Fortran::frontend
#endif // LLVM_FLANG_FRONTEND_COMPILERINVOCATION_H
