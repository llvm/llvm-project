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
#include "flang/Frontend/PreprocessorOptions.h"
#include "flang/Parser/parsing.h"
#include "flang/Semantics/semantics.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/DiagnosticOptions.h"
#include "llvm/Option/ArgList.h"
#include <memory>

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
  /// Options for the preprocessor.
  std::shared_ptr<Fortran::frontend::PreprocessorOptions> preprocessorOpts_;

  CompilerInvocationBase();
  CompilerInvocationBase(const CompilerInvocationBase &x);
  ~CompilerInvocationBase();

  clang::DiagnosticOptions &GetDiagnosticOpts() {
    return *diagnosticOpts_.get();
  }
  const clang::DiagnosticOptions &GetDiagnosticOpts() const {
    return *diagnosticOpts_.get();
  }

  PreprocessorOptions &preprocessorOpts() { return *preprocessorOpts_; }
  const PreprocessorOptions &preprocessorOpts() const {
    return *preprocessorOpts_;
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

  // Semantics context
  std::unique_ptr<Fortran::semantics::SemanticsContext> semanticsContext_;

  /// Semantic options
  // TODO: Merge with or translate to frontendOpts_. We shouldn't need two sets
  // of options.
  std::string moduleDir_ = ".";

  bool debugModuleDir_ = false;

  // Fortran Dialect options
  Fortran::common::IntrinsicTypeDefaultKinds defaultKinds_;

public:
  CompilerInvocation() = default;

  FrontendOptions &frontendOpts() { return frontendOpts_; }
  const FrontendOptions &frontendOpts() const { return frontendOpts_; }

  Fortran::parser::Options &fortranOpts() { return parserOpts_; }
  const Fortran::parser::Options &fortranOpts() const { return parserOpts_; }

  Fortran::semantics::SemanticsContext &semanticsContext() {
    return *semanticsContext_;
  }
  const Fortran::semantics::SemanticsContext &semanticsContext() const {
    return *semanticsContext_;
  }

  std::string &moduleDir() { return moduleDir_; }
  const std::string &moduleDir() const { return moduleDir_; }

  bool &debugModuleDir() { return debugModuleDir_; }
  const bool &debugModuleDir() const { return debugModuleDir_; }

  Fortran::common::IntrinsicTypeDefaultKinds &defaultKinds() {
    return defaultKinds_;
  }
  const Fortran::common::IntrinsicTypeDefaultKinds &defaultKinds() const {
    return defaultKinds_;
  }

  /// Create a compiler invocation from a list of input options.
  /// \returns true on success.
  /// \returns false if an error was encountered while parsing the arguments
  /// \param [out] res - The resulting invocation.
  static bool CreateFromArgs(CompilerInvocation &res,
      llvm::ArrayRef<const char *> commandLineArgs,
      clang::DiagnosticsEngine &diags);

  /// Useful setters
  void SetModuleDir(std::string &moduleDir) { moduleDir_ = moduleDir; }

  void SetDebugModuleDir(bool flag) { debugModuleDir_ = flag; }

  /// Set the Fortran options to predifined defaults. These defaults are
  /// consistend with f18/f18.cpp.
  // TODO: We should map frontendOpts_ to parserOpts_ instead. For that, we
  // need to extend frontendOpts_ first. Next, we need to add the corresponding
  // compiler driver options in libclangDriver.
  void SetDefaultFortranOpts();

  /// Set the default predefinitions.
  void setDefaultPredefinitions();

  /// Set the Fortran options to user-specified values.
  /// These values are found in the preprocessor options.
  void setFortranOpts();

  /// Set the Semantic Options
  void setSemanticsOpts(Fortran::parser::AllCookedSources &);
};

} // end namespace Fortran::frontend
#endif // LLVM_FLANG_FRONTEND_COMPILERINVOCATION_H
