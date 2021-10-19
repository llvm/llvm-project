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

  std::string moduleFileSuffix_ = ".mod";

  bool debugModuleDir_ = false;

  bool warnAsErr_ = false;

  /// This flag controls the unparsing and is used to decide whether to print out
  /// the semantically analyzed version of an object or expression or the plain
  /// version that does not include any information from semantic analysis.
  bool useAnalyzedObjectsForUnparse_ = true;

  // Fortran Dialect options
  Fortran::common::IntrinsicTypeDefaultKinds defaultKinds_;

  bool EnableConformanceChecks_ = false;

  /// Used in e.g. unparsing to dump the analyzed rather than the original
  /// parse-tree objects.
  Fortran::parser::AnalyzedObjectsAsFortran AsFortran_{
      [](llvm::raw_ostream &o, const Fortran::evaluate::GenericExprWrapper &x) {
        if (x.v) {
          x.v->AsFortran(o);
        } else {
          o << "(bad expression)";
        }
      },
      [](llvm::raw_ostream &o,
          const Fortran::evaluate::GenericAssignmentWrapper &x) {
        if (x.v) {
          x.v->AsFortran(o);
        } else {
          o << "(bad assignment)";
        }
      },
      [](llvm::raw_ostream &o, const Fortran::evaluate::ProcedureRef &x) {
        x.AsFortran(o << "CALL ");
      },
  };

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

  std::string &moduleFileSuffix() { return moduleFileSuffix_; }
  const std::string &moduleFileSuffix() const { return moduleFileSuffix_; }

  bool &debugModuleDir() { return debugModuleDir_; }
  const bool &debugModuleDir() const { return debugModuleDir_; }

  bool &warnAsErr() { return warnAsErr_; }
  const bool &warnAsErr() const { return warnAsErr_; }

  bool &useAnalyzedObjectsForUnparse() { return useAnalyzedObjectsForUnparse_; }
  const bool &useAnalyzedObjectsForUnparse() const {
    return useAnalyzedObjectsForUnparse_;
  }

  bool &enableConformanceChecks() { return EnableConformanceChecks_; }
  const bool &enableConformanceChecks() const {
    return EnableConformanceChecks_;
  }

  Fortran::parser::AnalyzedObjectsAsFortran &asFortran() { return AsFortran_; }
  const Fortran::parser::AnalyzedObjectsAsFortran &asFortran() const {
    return AsFortran_;
  }

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

  // Enables the std=f2018 conformance check
  void set_EnableConformanceChecks() { EnableConformanceChecks_ = true; }

  /// Useful setters
  void SetModuleDir(std::string &moduleDir) { moduleDir_ = moduleDir; }

  void SetModuleFileSuffix(const char *moduleFileSuffix) {
    moduleFileSuffix_ = std::string(moduleFileSuffix);
  }

  void SetDebugModuleDir(bool flag) { debugModuleDir_ = flag; }

  void SetWarnAsErr(bool flag) { warnAsErr_ = flag; }

  void SetUseAnalyzedObjectsForUnparse(bool flag) {
    useAnalyzedObjectsForUnparse_ = flag;
  }

  /// Set the Fortran options to predefined defaults.
  // TODO: We should map frontendOpts_ to parserOpts_ instead. For that, we
  // need to extend frontendOpts_ first. Next, we need to add the corresponding
  // compiler driver options in libclangDriver.
  void SetDefaultFortranOpts();

  /// Set the default predefinitions.
  void setDefaultPredefinitions();

  /// Collect the macro definitions from preprocessorOpts_ and prepare them for
  /// the parser (i.e. copy into parserOpts_)
  void collectMacroDefinitions();

  /// Set the Fortran options to user-specified values.
  /// These values are found in the preprocessor options.
  void setFortranOpts();

  /// Set the Semantic Options
  void setSemanticsOpts(Fortran::parser::AllCookedSources &);
};

} // end namespace Fortran::frontend
#endif // LLVM_FLANG_FRONTEND_COMPILERINVOCATION_H
