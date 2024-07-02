//===- CompilerInvocation.h - Compiler Invocation Helper Data ---*- C -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Coding style: https://mlir.llvm.org/getting_started/DeveloperGuide/
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_FRONTEND_COMPILERINVOCATION_H
#define FORTRAN_FRONTEND_COMPILERINVOCATION_H

#include "flang/Frontend/CodeGenOptions.h"
#include "flang/Frontend/FrontendOptions.h"
#include "flang/Frontend/LangOptions.h"
#include "flang/Frontend/PreprocessorOptions.h"
#include "flang/Frontend/TargetOptions.h"
#include "flang/Lower/LoweringOptions.h"
#include "flang/Parser/parsing.h"
#include "flang/Semantics/semantics.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/DiagnosticOptions.h"
#include "llvm/Option/ArgList.h"
#include <memory>

namespace llvm {
class TargetMachine;
}

namespace Fortran::frontend {

/// Fill out Opts based on the options given in Args.
///
/// When errors are encountered, return false and, if Diags is non-null,
/// report the error(s).
bool parseDiagnosticArgs(clang::DiagnosticOptions &opts,
                         llvm::opt::ArgList &args);

class CompilerInvocationBase {
public:
  /// Options controlling the diagnostic engine.
  llvm::IntrusiveRefCntPtr<clang::DiagnosticOptions> diagnosticOpts;
  /// Options for the preprocessor.
  std::shared_ptr<Fortran::frontend::PreprocessorOptions> preprocessorOpts;

  CompilerInvocationBase();
  CompilerInvocationBase(const CompilerInvocationBase &x);
  ~CompilerInvocationBase();

  clang::DiagnosticOptions &getDiagnosticOpts() {
    return *diagnosticOpts.get();
  }
  const clang::DiagnosticOptions &getDiagnosticOpts() const {
    return *diagnosticOpts.get();
  }

  PreprocessorOptions &getPreprocessorOpts() { return *preprocessorOpts; }
  const PreprocessorOptions &getPreprocessorOpts() const {
    return *preprocessorOpts;
  }
};

class CompilerInvocation : public CompilerInvocationBase {
  /// Options for the frontend driver
  // TODO: Merge with or translate to parserOpts_. We shouldn't need two sets of
  // options.
  FrontendOptions frontendOpts;

  /// Options for Flang parser
  // TODO: Merge with or translate to frontendOpts. We shouldn't need two sets
  // of options.
  Fortran::parser::Options parserOpts;

  /// Options controlling lowering.
  Fortran::lower::LoweringOptions loweringOpts;

  /// Options controlling the target.
  Fortran::frontend::TargetOptions targetOpts;

  /// Options controlling IRgen and the backend.
  Fortran::frontend::CodeGenOptions codeGenOpts;

  /// Options controlling language dialect.
  Fortran::frontend::LangOptions langOpts;

  // The original invocation of the compiler driver.
  // This string will be set as the return value from the COMPILER_OPTIONS
  // intrinsic of iso_fortran_env.
  std::string allCompilerInvocOpts;

  /// Semantic options
  // TODO: Merge with or translate to frontendOpts. We shouldn't need two sets
  // of options.
  std::string moduleDir = ".";

  std::string moduleFileSuffix = ".mod";

  bool debugModuleDir = false;

  bool warnAsErr = false;

  // Executable name
  const char *argv0;

  /// This flag controls the unparsing and is used to decide whether to print
  /// out the semantically analyzed version of an object or expression or the
  /// plain version that does not include any information from semantic
  /// analysis.
  bool useAnalyzedObjectsForUnparse = true;

  // Fortran Dialect options
  Fortran::common::IntrinsicTypeDefaultKinds defaultKinds;

  // Fortran Warning options
  bool enableConformanceChecks = false;
  bool enableUsageChecks = false;
  bool disableWarnings = false;

  /// Used in e.g. unparsing to dump the analyzed rather than the original
  /// parse-tree objects.
  Fortran::parser::AnalyzedObjectsAsFortran asFortran{
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

  FrontendOptions &getFrontendOpts() { return frontendOpts; }
  const FrontendOptions &getFrontendOpts() const { return frontendOpts; }

  Fortran::parser::Options &getFortranOpts() { return parserOpts; }
  const Fortran::parser::Options &getFortranOpts() const { return parserOpts; }

  TargetOptions &getTargetOpts() { return targetOpts; }
  const TargetOptions &getTargetOpts() const { return targetOpts; }

  CodeGenOptions &getCodeGenOpts() { return codeGenOpts; }
  const CodeGenOptions &getCodeGenOpts() const { return codeGenOpts; }

  LangOptions &getLangOpts() { return langOpts; }
  const LangOptions &getLangOpts() const { return langOpts; }

  Fortran::lower::LoweringOptions &getLoweringOpts() { return loweringOpts; }
  const Fortran::lower::LoweringOptions &getLoweringOpts() const {
    return loweringOpts;
  }

  /// Creates and configures semantics context based on the compilation flags.
  std::unique_ptr<Fortran::semantics::SemanticsContext>
  getSemanticsCtx(Fortran::parser::AllCookedSources &allCookedSources,
                  const llvm::TargetMachine &);

  std::string &getModuleDir() { return moduleDir; }
  const std::string &getModuleDir() const { return moduleDir; }

  std::string &getModuleFileSuffix() { return moduleFileSuffix; }
  const std::string &getModuleFileSuffix() const { return moduleFileSuffix; }

  bool &getDebugModuleDir() { return debugModuleDir; }
  const bool &getDebugModuleDir() const { return debugModuleDir; }

  bool &getWarnAsErr() { return warnAsErr; }
  const bool &getWarnAsErr() const { return warnAsErr; }

  bool &getUseAnalyzedObjectsForUnparse() {
    return useAnalyzedObjectsForUnparse;
  }
  const bool &getUseAnalyzedObjectsForUnparse() const {
    return useAnalyzedObjectsForUnparse;
  }

  bool &getEnableConformanceChecks() { return enableConformanceChecks; }
  const bool &getEnableConformanceChecks() const {
    return enableConformanceChecks;
  }

  const char *getArgv0() { return argv0; }

  bool &getEnableUsageChecks() { return enableUsageChecks; }
  const bool &getEnableUsageChecks() const { return enableUsageChecks; }

  bool &getDisableWarnings() { return disableWarnings; }
  const bool &getDisableWarnings() const { return disableWarnings; }

  Fortran::parser::AnalyzedObjectsAsFortran &getAsFortran() {
    return asFortran;
  }
  const Fortran::parser::AnalyzedObjectsAsFortran &getAsFortran() const {
    return asFortran;
  }

  Fortran::common::IntrinsicTypeDefaultKinds &getDefaultKinds() {
    return defaultKinds;
  }
  const Fortran::common::IntrinsicTypeDefaultKinds &getDefaultKinds() const {
    return defaultKinds;
  }

  /// Create a compiler invocation from a list of input options.
  /// \returns true on success.
  /// \returns false if an error was encountered while parsing the arguments
  /// \param [out] res - The resulting invocation.
  static bool createFromArgs(CompilerInvocation &res,
                             llvm::ArrayRef<const char *> commandLineArgs,
                             clang::DiagnosticsEngine &diags,
                             const char *argv0 = nullptr);

  // Enables the std=f2018 conformance check
  void setEnableConformanceChecks() { enableConformanceChecks = true; }

  // Enables the usage checks
  void setEnableUsageChecks() { enableUsageChecks = true; }

  // Disables all Warnings
  void setDisableWarnings() { disableWarnings = true; }

  /// Useful setters
  void setArgv0(const char *dir) { argv0 = dir; }

  void setModuleDir(std::string &dir) { moduleDir = dir; }

  void setModuleFileSuffix(const char *suffix) {
    moduleFileSuffix = std::string(suffix);
  }

  void setDebugModuleDir(bool flag) { debugModuleDir = flag; }

  void setWarnAsErr(bool flag) { warnAsErr = flag; }

  void setUseAnalyzedObjectsForUnparse(bool flag) {
    useAnalyzedObjectsForUnparse = flag;
  }

  /// Set the Fortran options to predefined defaults.
  // TODO: We should map frontendOpts_ to parserOpts_ instead. For that, we
  // need to extend frontendOpts_ first. Next, we need to add the corresponding
  // compiler driver options in libclangDriver.
  void setDefaultFortranOpts();

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

  /// Set \p loweringOptions controlling lowering behavior based
  /// on the \p optimizationLevel.
  void setLoweringOptions();
};

} // end namespace Fortran::frontend
#endif // FORTRAN_FRONTEND_COMPILERINVOCATION_H
