//===-- CoverageChecker.h - Module map coverage checker -*- C++ -*-------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===--------------------------------------------------------------------===//
///
/// \file
/// Definitions for CoverageChecker.
///
//===--------------------------------------------------------------------===//

#ifndef COVERAGECHECKER_H
#define COVERAGECHECKER_H

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
#include "llvm/ADT/StringSet.h"
#include "llvm/TargetParser/Host.h"
#include <string>
#include <vector>

namespace Modularize {

/// Module map checker class.
/// This is the heart of the checker.
/// The doChecks function does the main work.
/// The data members store the options and internally collected data.
class CoverageChecker {
  // Checker arguments.

  /// The module.modulemap file path. Can be relative or absolute.
  llvm::StringRef ModuleMapPath;
  /// The include paths to check for files.
  /// (Note that other directories above these paths are ignored.
  /// To expect all files to be accounted for from the module.modulemap
  /// file directory on down, leave this empty.)
  std::vector<std::string> IncludePaths;
  /// The remaining arguments, to be passed to the front end.
  llvm::ArrayRef<std::string> CommandLine;
  /// The module map.
  clang::ModuleMap *ModMap;

  // Internal data.

  /// Directory containing the module map.
  /// Might be relative to the current directory, or absolute.
  std::string ModuleMapDirectory;
  /// Set of all the headers found in the module map.
  llvm::StringSet<llvm::MallocAllocator> ModuleMapHeadersSet;
  /// All the headers found in the file system starting at the
  /// module map, or the union of those from the include paths.
  std::vector<std::string> FileSystemHeaders;
  /// Headers found in file system, but not in module map.
  std::vector<std::string> UnaccountedForHeaders;

public:
  /// Constructor.
  /// You can use the static createCoverageChecker to create an instance
  /// of this object.
  /// \param ModuleMapPath The module.modulemap file path.
  ///   Can be relative or absolute.
  /// \param IncludePaths The include paths to check for files.
  ///   (Note that other directories above these paths are ignored.
  ///   To expect all files to be accounted for from the module.modulemap
  ///   file directory on down, leave this empty.)
  /// \param CommandLine Compile command line arguments.
  /// \param ModuleMap The module map to check.
  CoverageChecker(llvm::StringRef ModuleMapPath,
    std::vector<std::string> &IncludePaths,
    llvm::ArrayRef<std::string> CommandLine,
    clang::ModuleMap *ModuleMap);

  /// Create instance of CoverageChecker.
  /// \param ModuleMapPath The module.modulemap file path.
  ///   Can be relative or absolute.
  /// \param IncludePaths The include paths to check for files.
  ///   (Note that other directories above these paths are ignored.
  ///   To expect all files to be accounted for from the module.modulemap
  ///   file directory on down, leave this empty.)
  /// \param CommandLine Compile command line arguments.
  /// \param ModuleMap The module map to check.
  /// \returns Initialized CoverageChecker object.
  static std::unique_ptr<CoverageChecker> createCoverageChecker(
      llvm::StringRef ModuleMapPath, std::vector<std::string> &IncludePaths,
      llvm::ArrayRef<std::string> CommandLine, clang::ModuleMap *ModuleMap);

  /// Do checks.
  /// Starting from the directory of the module.modulemap file,
  /// Find all header files, optionally looking only at files
  /// covered by the include path options, and compare against
  /// the headers referenced by the module.modulemap file.
  /// Display warnings for unaccounted-for header files.
  /// \returns 0 if there were no errors or warnings, 1 if there
  ///   were warnings, 2 if any other problem, such as a bad
  ///   module map path argument was specified.
  std::error_code doChecks();

  // The following functions are called by doChecks.

  /// Collect module headers.
  /// Walks the modules and collects referenced headers into
  /// ModuleMapHeadersSet.
  void collectModuleHeaders();

  /// Collect referenced headers from one module.
  /// Collects the headers referenced in the given module into
  /// ModuleMapHeadersSet.
  /// \param Mod The module reference.
  /// \return True if no errors.
  bool collectModuleHeaders(const clang::Module &Mod);

  /// Collect headers from an umbrella directory.
  /// \param UmbrellaDirName The umbrella directory name.
  /// \return True if no errors.
  bool collectUmbrellaHeaders(llvm::StringRef UmbrellaDirName);

  /// Collect headers referenced from an umbrella file.
  /// \param UmbrellaHeaderName The umbrella file path.
  /// \return True if no errors.
  bool collectUmbrellaHeaderHeaders(llvm::StringRef UmbrellaHeaderName);

  /// Called from CoverageCheckerCallbacks to track a header included
  /// from an umbrella header.
  /// \param HeaderName The header file path.
  void collectUmbrellaHeaderHeader(llvm::StringRef HeaderName);

  /// Collect file system header files.
  /// This function scans the file system for header files,
  /// starting at the directory of the module.modulemap file,
  /// optionally filtering out all but the files covered by
  /// the include path options.
  /// \returns True if no errors.
  bool collectFileSystemHeaders();

  /// Collect file system header files from the given path.
  /// This function scans the file system for header files,
  /// starting at the given directory, which is assumed to be
  /// relative to the directory of the module.modulemap file.
  /// \returns True if no errors.
  bool collectFileSystemHeaders(llvm::StringRef IncludePath);

  /// Find headers unaccounted-for in module map.
  /// This function compares the list of collected header files
  /// against those referenced in the module map.  Display
  /// warnings for unaccounted-for header files.
  /// Save unaccounted-for file list for possible.
  /// fixing action.
  void findUnaccountedForHeaders();
};

} // end namespace Modularize

#endif // COVERAGECHECKER_H
