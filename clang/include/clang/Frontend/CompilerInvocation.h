//===- CompilerInvocation.h - Compiler Invocation Helper Data ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_FRONTEND_COMPILERINVOCATION_H
#define LLVM_CLANG_FRONTEND_COMPILERINVOCATION_H

#include "clang/Basic/CodeGenOptions.h"
#include "clang/Basic/DiagnosticOptions.h"
#include "clang/Basic/FileSystemOptions.h"
#include "clang/Basic/LLVM.h"
#include "clang/Basic/LangOptions.h"
#include "clang/Basic/LangStandard.h"
#include "clang/Frontend/DependencyOutputOptions.h"
#include "clang/Frontend/FrontendOptions.h"
#include "clang/Frontend/MigratorOptions.h"
#include "clang/Frontend/PreprocessorOutputOptions.h"
#include "clang/StaticAnalyzer/Core/AnalyzerOptions.h"
#include "llvm/ADT/IntrusiveRefCntPtr.h"
#include "llvm/ADT/ArrayRef.h"
#include <memory>
#include <string>

namespace llvm {

class Triple;

namespace opt {

class ArgList;

} // namespace opt

namespace vfs {

class FileSystem;

} // namespace vfs

} // namespace llvm

namespace clang {

class DiagnosticsEngine;
class HeaderSearchOptions;
class PreprocessorOptions;
class TargetOptions;

// This lets us create the DiagnosticsEngine with a properly-filled-out
// DiagnosticOptions instance.
std::unique_ptr<DiagnosticOptions>
CreateAndPopulateDiagOpts(ArrayRef<const char *> Argv);

/// Fill out Opts based on the options given in Args.
///
/// Args must have been created from the OptTable returned by
/// createCC1OptTable().
///
/// When errors are encountered, return false and, if Diags is non-null,
/// report the error(s).
bool ParseDiagnosticArgs(DiagnosticOptions &Opts, llvm::opt::ArgList &Args,
                         DiagnosticsEngine *Diags = nullptr,
                         bool DefaultDiagColor = true);

/// The base class of CompilerInvocation with reference semantics.
///
/// This class stores option objects behind reference-counted pointers. This is
/// useful for clients that want to keep some option object around even after
/// CompilerInvocation gets destroyed, without making a copy.
///
/// This is a separate class so that we can implement the copy constructor and
/// assignment here and leave them defaulted in the rest of CompilerInvocation.
class CompilerInvocationRefBase {
public:
  /// Options controlling the language variant.
  std::shared_ptr<LangOptions> LangOpts;

  /// Options controlling the target.
  std::shared_ptr<TargetOptions> TargetOpts;

  /// Options controlling the diagnostic engine.
  IntrusiveRefCntPtr<DiagnosticOptions> DiagnosticOpts;

  /// Options controlling the \#include directive.
  std::shared_ptr<HeaderSearchOptions> HeaderSearchOpts;

  /// Options controlling the preprocessor (aside from \#include handling).
  std::shared_ptr<PreprocessorOptions> PreprocessorOpts;

  /// Options controlling the static analyzer.
  AnalyzerOptionsRef AnalyzerOpts;

  CompilerInvocationRefBase();
  CompilerInvocationRefBase(const CompilerInvocationRefBase &X);
  CompilerInvocationRefBase(CompilerInvocationRefBase &&X);
  CompilerInvocationRefBase &operator=(CompilerInvocationRefBase X);
  CompilerInvocationRefBase &operator=(CompilerInvocationRefBase &&X);
  ~CompilerInvocationRefBase();

  LangOptions *getLangOpts() { return LangOpts.get(); }
  const LangOptions *getLangOpts() const { return LangOpts.get(); }

  TargetOptions &getTargetOpts() { return *TargetOpts.get(); }
  const TargetOptions &getTargetOpts() const { return *TargetOpts.get(); }

  DiagnosticOptions &getDiagnosticOpts() const { return *DiagnosticOpts; }

  HeaderSearchOptions &getHeaderSearchOpts() { return *HeaderSearchOpts; }

  const HeaderSearchOptions &getHeaderSearchOpts() const {
    return *HeaderSearchOpts;
  }

  std::shared_ptr<HeaderSearchOptions> getHeaderSearchOptsPtr() const {
    return HeaderSearchOpts;
  }

  std::shared_ptr<PreprocessorOptions> getPreprocessorOptsPtr() {
    return PreprocessorOpts;
  }

  PreprocessorOptions &getPreprocessorOpts() { return *PreprocessorOpts; }

  const PreprocessorOptions &getPreprocessorOpts() const {
    return *PreprocessorOpts;
  }

  AnalyzerOptionsRef getAnalyzerOpts() const { return AnalyzerOpts; }
};

/// The base class of CompilerInvocation with value semantics.
class CompilerInvocationValueBase {
protected:
  MigratorOptions MigratorOpts;

  /// Options controlling IRgen and the backend.
  CodeGenOptions CodeGenOpts;

  /// Options controlling dependency output.
  DependencyOutputOptions DependencyOutputOpts;

  /// Options controlling file system operations.
  FileSystemOptions FileSystemOpts;

  /// Options controlling the frontend itself.
  FrontendOptions FrontendOpts;

  /// Options controlling preprocessed output.
  PreprocessorOutputOptions PreprocessorOutputOpts;

public:
  MigratorOptions &getMigratorOpts() { return MigratorOpts; }
  const MigratorOptions &getMigratorOpts() const { return MigratorOpts; }

  CodeGenOptions &getCodeGenOpts() { return CodeGenOpts; }
  const CodeGenOptions &getCodeGenOpts() const { return CodeGenOpts; }

  DependencyOutputOptions &getDependencyOutputOpts() {
    return DependencyOutputOpts;
  }

  const DependencyOutputOptions &getDependencyOutputOpts() const {
    return DependencyOutputOpts;
  }

  FileSystemOptions &getFileSystemOpts() { return FileSystemOpts; }

  const FileSystemOptions &getFileSystemOpts() const {
    return FileSystemOpts;
  }

  FrontendOptions &getFrontendOpts() { return FrontendOpts; }
  const FrontendOptions &getFrontendOpts() const { return FrontendOpts; }

  PreprocessorOutputOptions &getPreprocessorOutputOpts() {
    return PreprocessorOutputOpts;
  }

  const PreprocessorOutputOptions &getPreprocessorOutputOpts() const {
    return PreprocessorOutputOpts;
  }
};

/// Helper class for holding the data necessary to invoke the compiler.
///
/// This class is designed to represent an abstract "invocation" of the
/// compiler, including data such as the include paths, the code generation
/// options, the warning flags, and so on.
class CompilerInvocation : public CompilerInvocationRefBase,
                           public CompilerInvocationValueBase {
public:
  /// Create a compiler invocation from a list of input options.
  /// \returns true on success.
  ///
  /// \returns false if an error was encountered while parsing the arguments
  /// and attempts to recover and continue parsing the rest of the arguments.
  /// The recovery is best-effort and only guarantees that \p Res will end up in
  /// one of the vaild-to-access (albeit arbitrary) states.
  ///
  /// \param [out] Res - The resulting invocation.
  /// \param [in] CommandLineArgs - Array of argument strings, this must not
  /// contain "-cc1".
  static bool CreateFromArgs(CompilerInvocation &Res,
                             ArrayRef<const char *> CommandLineArgs,
                             DiagnosticsEngine &Diags,
                             const char *Argv0 = nullptr);

  /// Get the directory where the compiler headers
  /// reside, relative to the compiler binary (found by the passed in
  /// arguments).
  ///
  /// \param Argv0 - The program path (from argv[0]), for finding the builtin
  /// compiler path.
  /// \param MainAddr - The address of main (or some other function in the main
  /// executable), for finding the builtin compiler path.
  static std::string GetResourcesPath(const char *Argv0, void *MainAddr);

  /// Retrieve a module hash string that is suitable for uniquely
  /// identifying the conditions under which the module was built.
  std::string getModuleHash() const;

  using StringAllocator = llvm::function_ref<const char *(const llvm::Twine &)>;
  /// Generate cc1-compatible command line arguments from this instance.
  ///
  /// \param [out] Args - The generated arguments. Note that the caller is
  /// responsible for inserting the path to the clang executable and "-cc1" if
  /// desired.
  /// \param SA - A function that given a Twine can allocate storage for a given
  /// command line argument and return a pointer to the newly allocated string.
  /// The returned pointer is what gets appended to Args.
  void generateCC1CommandLine(llvm::SmallVectorImpl<const char *> &Args,
                              StringAllocator SA) const;

  /// Generate cc1-compatible command line arguments from this instance,
  /// wrapping the result as a std::vector<std::string>.
  ///
  /// This is a (less-efficient) wrapper over generateCC1CommandLine().
  std::vector<std::string> getCC1CommandLine() const;

  /// Check that \p Args can be parsed and re-serialized without change,
  /// emiting diagnostics for any differences.
  ///
  /// This check is only suitable for command-lines that are expected to already
  /// be canonical.
  ///
  /// \return false if there are any errors.
  static bool checkCC1RoundTrip(ArrayRef<const char *> Args,
                                DiagnosticsEngine &Diags,
                                const char *Argv0 = nullptr);

  /// Reset all of the options that are not considered when building a
  /// module.
  void resetNonModularOptions();

  /// Disable implicit modules and canonicalize options that are only used by
  /// implicit modules.
  void clearImplicitModuleBuildOptions();

private:
  static bool CreateFromArgsImpl(CompilerInvocation &Res,
                                 ArrayRef<const char *> CommandLineArgs,
                                 DiagnosticsEngine &Diags, const char *Argv0);

  /// Generate command line options from DiagnosticOptions.
  static void GenerateDiagnosticArgs(const DiagnosticOptions &Opts,
                                     SmallVectorImpl<const char *> &Args,
                                     StringAllocator SA, bool DefaultDiagColor);

  /// Parse command line options that map to LangOptions.
  static bool ParseLangArgs(LangOptions &Opts, llvm::opt::ArgList &Args,
                            InputKind IK, const llvm::Triple &T,
                            std::vector<std::string> &Includes,
                            DiagnosticsEngine &Diags);

  /// Generate command line options from LangOptions.
  static void GenerateLangArgs(const LangOptions &Opts,
                               SmallVectorImpl<const char *> &Args,
                               StringAllocator SA, const llvm::Triple &T,
                               InputKind IK);

  /// Parse command line options that map to CodeGenOptions.
  static bool ParseCodeGenArgs(CodeGenOptions &Opts, llvm::opt::ArgList &Args,
                               InputKind IK, DiagnosticsEngine &Diags,
                               const llvm::Triple &T,
                               const std::string &OutputFile,
                               const LangOptions &LangOptsRef);

  // Generate command line options from CodeGenOptions.
  static void GenerateCodeGenArgs(const CodeGenOptions &Opts,
                                  SmallVectorImpl<const char *> &Args,
                                  StringAllocator SA, const llvm::Triple &T,
                                  const std::string &OutputFile,
                                  const LangOptions *LangOpts);
};

IntrusiveRefCntPtr<llvm::vfs::FileSystem>
createVFSFromCompilerInvocation(const CompilerInvocation &CI,
                                DiagnosticsEngine &Diags);

IntrusiveRefCntPtr<llvm::vfs::FileSystem> createVFSFromCompilerInvocation(
    const CompilerInvocation &CI, DiagnosticsEngine &Diags,
    IntrusiveRefCntPtr<llvm::vfs::FileSystem> BaseFS);

IntrusiveRefCntPtr<llvm::vfs::FileSystem>
createVFSFromOverlayFiles(ArrayRef<std::string> VFSOverlayFiles,
                          DiagnosticsEngine &Diags,
                          IntrusiveRefCntPtr<llvm::vfs::FileSystem> BaseFS);

} // namespace clang

#endif // LLVM_CLANG_FRONTEND_COMPILERINVOCATION_H
