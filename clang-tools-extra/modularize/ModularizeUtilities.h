//=====-- ModularizeUtilities.h - Utilities for modularize -*- C++ -*-======//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===--------------------------------------------------------------------===//
///
/// \file
/// ModularizeUtilities class definition.
///
//===--------------------------------------------------------------------===//

#ifndef MODULARIZEUTILITIES_H
#define MODULARIZEUTILITIES_H

#include "Modularize.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/LangOptions.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Basic/TargetOptions.h"
#include "clang/Frontend/TextDiagnosticPrinter.h"
#include "clang/Lex/HeaderSearch.h"
#include "clang/Lex/HeaderSearchOptions.h"
#include "clang/Lex/ModuleMap.h"
#include "clang/Lex/Preprocessor.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSet.h"
#include <string>
#include <vector>

namespace Modularize {

/// Modularize utilities class.
/// Support functions and data for modularize.
class ModularizeUtilities {
public:
  // Input arguments.

  /// The input file paths.
  std::vector<std::string> InputFilePaths;
  /// The header prefix.
  llvm::StringRef HeaderPrefix;
  /// The path of problem files list file.
  llvm::StringRef ProblemFilesPath;

  // Output data.

  /// List of top-level header files.
  llvm::SmallVector<std::string, 32> HeaderFileNames;
  /// Map of top-level header file dependencies.
  DependencyMap Dependencies;
  /// True if we have module maps.
  bool HasModuleMap;
  /// Missing header count.
  int MissingHeaderCount;
  /// List of header files with no problems during the first pass,
  /// that is, no compile errors.
  llvm::SmallVector<std::string, 32> GoodFileNames;
  /// List of header files with problems.
  llvm::SmallVector<std::string, 32> ProblemFileNames;

  // Functions.

  /// Constructor.
  /// You can use the static createModularizeUtilities to create an instance
  /// of this object.
  /// \param InputPaths The input file paths.
  /// \param Prefix The headear path prefix.
  /// \param ProblemFilesListPath The problem header list path.
  ModularizeUtilities(std::vector<std::string> &InputPaths,
                      llvm::StringRef Prefix,
                      llvm::StringRef ProblemFilesListPath);

  /// Create instance of ModularizeUtilities.
  /// \param InputPaths The input file paths.
  /// \param Prefix The headear path prefix.
  /// \param ProblemFilesListPath The problem header list path.
  /// \returns Initialized ModularizeUtilities object.
  static ModularizeUtilities *createModularizeUtilities(
      std::vector<std::string> &InputPaths,
      llvm::StringRef Prefix,
      llvm::StringRef ProblemFilesListPath);

  /// Load header list and dependencies.
  /// \returns std::error_code.
  std::error_code loadAllHeaderListsAndDependencies();

  /// Do coverage checks.
  /// For each loaded module map, do header coverage check.
  /// Starting from the directory of the module.modulemap file,
  /// Find all header files, optionally looking only at files
  /// covered by the include path options, and compare against
  /// the headers referenced by the module.modulemap file.
  /// Display warnings for unaccounted-for header files.
  /// \param IncludePaths The include paths to check for files.
  ///   (Note that other directories above these paths are ignored.
  ///   To expect all files to be accounted for from the module.modulemap
  ///   file directory on down, leave this empty.)
  /// \param CommandLine Compile command line arguments.
  /// \returns 0 if there were no errors or warnings, 1 if there
  ///   were warnings, 2 if any other problem, such as a bad
  ///   module map path argument was specified.
  std::error_code doCoverageCheck(std::vector<std::string> &IncludePaths,
                                  llvm::ArrayRef<std::string> CommandLine);

  /// Add unique problem file.
  /// Also standardizes the path.
  /// \param FilePath Problem file path.
  void addUniqueProblemFile(std::string FilePath);

  /// Add file with no compile errors.
  /// Also standardizes the path.
  /// \param FilePath Problem file path.
  void addNoCompileErrorsFile(std::string FilePath);

  /// List problem files.
  void displayProblemFiles();

  /// List files with no problems.
  void displayGoodFiles();

  /// List files with problem files commented out.
  void displayCombinedFiles();

  // Internal.

protected:

  /// Load single header list and dependencies.
  /// \param InputPath The input file path.
  /// \returns std::error_code.
  std::error_code loadSingleHeaderListsAndDependencies(
      llvm::StringRef InputPath);

  /// Load problem header list.
  /// \param InputPath The input file path.
  /// \returns std::error_code.
  std::error_code loadProblemHeaderList(
    llvm::StringRef InputPath);

  /// Load single module map and extract header file list.
  /// \param InputPath The input file path.
  /// \returns std::error_code.
  std::error_code loadModuleMap(
    llvm::StringRef InputPath);

  /// Collect module Map headers.
  /// Walks the modules and collects referenced headers into
  /// HeaderFileNames.
  /// \param ModMap A loaded module map object.
  /// \return True if no errors.
  bool collectModuleMapHeaders(clang::ModuleMap *ModMap);

  /// Collect referenced headers from one module.
  /// Collects the headers referenced in the given module into
  /// HeaderFileNames.
  /// \param Mod The module reference.
  /// \return True if no errors.
  bool collectModuleHeaders(const clang::Module &Mod);

  /// Collect headers from an umbrella directory.
  /// \param UmbrellaDirName The umbrella directory name.
  /// \return True if no errors.
  bool collectUmbrellaHeaders(llvm::StringRef UmbrellaDirName,
    DependentsVector &Dependents);

public:

  // Utility functions.

  /// Convert header path to canonical form.
  /// The canonical form is basically just use forward slashes,
  /// and remove "./".
  /// \param FilePath The file path.
  /// \returns The file path in canonical form.
  static std::string getCanonicalPath(llvm::StringRef FilePath);

  /// Check for header file extension.
  /// If the file extension is .h, .inc, or missing, it's
  /// assumed to be a header.
  /// \param FileName The file name.  Must not be a directory.
  /// \returns true if it has a header extension or no extension.
  static bool isHeader(llvm::StringRef FileName);

  /// Get directory path component from file path.
  /// \returns the component of the given path, which will be
  /// relative if the given path is relative, absolute if the
  /// given path is absolute, or "." if the path has no leading
  /// path component.
  static std::string getDirectoryFromPath(llvm::StringRef Path);

  // Internal data.

  /// Options controlling the language variant.
  std::shared_ptr<clang::LangOptions> LangOpts;
  /// Diagnostic IDs.
  const llvm::IntrusiveRefCntPtr<clang::DiagnosticIDs> DiagIDs;
  /// Options controlling the diagnostic engine.
  llvm::IntrusiveRefCntPtr<clang::DiagnosticOptions> DiagnosticOpts;
  /// Diagnostic consumer.
  clang::TextDiagnosticPrinter DC;
  /// Diagnostic engine.
  llvm::IntrusiveRefCntPtr<clang::DiagnosticsEngine> Diagnostics;
  /// Options controlling the target.
  std::shared_ptr<clang::TargetOptions> TargetOpts;
  /// Target information.
  llvm::IntrusiveRefCntPtr<clang::TargetInfo> Target;
  /// Options controlling the file system manager.
  clang::FileSystemOptions FileSystemOpts;
  /// File system manager.
  llvm::IntrusiveRefCntPtr<clang::FileManager> FileMgr;
  /// Source manager.
  llvm::IntrusiveRefCntPtr<clang::SourceManager> SourceMgr;
  /// Header search options.
  clang::HeaderSearchOptions HSOpts;
  /// Header search manager.
  std::unique_ptr<clang::HeaderSearch> HeaderInfo;
  // The loaded module map objects.
  std::vector<std::unique_ptr<clang::ModuleMap>> ModuleMaps;
};

} // end namespace Modularize

#endif // MODULARIZEUTILITIES_H
