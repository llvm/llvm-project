//===--- extra/module-map-checker/CoverageChecker.cpp -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a class that validates a module map by checking that
// all headers in the corresponding directories are accounted for.
//
// This class uses a previously loaded module map object.
// Starting at the module map file directory, or just the include
// paths, if specified, it will collect the names of all the files it
// considers headers (no extension, .h, or .inc--if you need more, modify the
// ModularizeUtilities::isHeader function).
//  It then compares the headers against those referenced
// in the module map, either explicitly named, or implicitly named via an
// umbrella directory or umbrella file, as parsed by the ModuleMap object.
// If headers are found which are not referenced or covered by an umbrella
// directory or file, warning messages will be produced, and the doChecks
// function will return an error code of 1.  Other errors result in an error
// code of 2. If no problems are found, an error code of 0 is returned.
//
// Note that in the case of umbrella headers, this tool invokes the compiler
// to preprocess the file, and uses a callback to collect the header files
// included by the umbrella header or any of its nested includes.  If any
// front end options are needed for these compiler invocations, these are
// to be passed in via the CommandLine parameter.
//
// Warning message have the form:
//
//  warning: module.modulemap does not account for file: Level3A.h
//
// Note that for the case of the module map referencing a file that does
// not exist, the module map parser in Clang will (at the time of this
// writing) display an error message.
//
// Potential problems with this program:
//
// 1. Might need a better header matching mechanism, or extensions to the
//    canonical file format used.
//
// 2. It might need to support additional header file extensions.
//
// Future directions:
//
// 1. Add an option to fix the problems found, writing a new module map.
//    Include an extra option to add unaccounted-for headers as excluded.
//
//===----------------------------------------------------------------------===//

#include "ModularizeUtilities.h"
#include "clang/AST/ASTConsumer.h"
#include "CoverageChecker.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Driver/Options.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendAction.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Lex/PPCallbacks.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Tooling/CompilationDatabase.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/Option/Option.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

using namespace Modularize;
using namespace clang;
using namespace clang::driver;
using namespace clang::driver::options;
using namespace clang::tooling;
namespace cl = llvm::cl;
namespace sys = llvm::sys;

// Preprocessor callbacks.
// We basically just collect include files.
class CoverageCheckerCallbacks : public PPCallbacks {
public:
  CoverageCheckerCallbacks(CoverageChecker &Checker) : Checker(Checker) {}
  ~CoverageCheckerCallbacks() override {}

  // Include directive callback.
  void InclusionDirective(SourceLocation HashLoc, const Token &IncludeTok,
                          StringRef FileName, bool IsAngled,
                          CharSourceRange FilenameRange,
                          OptionalFileEntryRef File, StringRef SearchPath,
                          StringRef RelativePath, const Module *SuggestedModule,
                          bool ModuleImported,
                          SrcMgr::CharacteristicKind FileType) override {
    Checker.collectUmbrellaHeaderHeader(File->getName());
  }

private:
  CoverageChecker &Checker;
};

// Frontend action stuff:

// Consumer is responsible for setting up the callbacks.
class CoverageCheckerConsumer : public ASTConsumer {
public:
  CoverageCheckerConsumer(CoverageChecker &Checker, Preprocessor &PP) {
    // PP takes ownership.
    PP.addPPCallbacks(std::make_unique<CoverageCheckerCallbacks>(Checker));
  }
};

class CoverageCheckerAction : public SyntaxOnlyAction {
public:
  CoverageCheckerAction(CoverageChecker &Checker) : Checker(Checker) {}

protected:
  std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI,
    StringRef InFile) override {
    return std::make_unique<CoverageCheckerConsumer>(Checker,
      CI.getPreprocessor());
  }

private:
  CoverageChecker &Checker;
};

class CoverageCheckerFrontendActionFactory : public FrontendActionFactory {
public:
  CoverageCheckerFrontendActionFactory(CoverageChecker &Checker)
    : Checker(Checker) {}

  std::unique_ptr<FrontendAction> create() override {
    return std::make_unique<CoverageCheckerAction>(Checker);
  }

private:
  CoverageChecker &Checker;
};

// CoverageChecker class implementation.

// Constructor.
CoverageChecker::CoverageChecker(StringRef ModuleMapPath,
    std::vector<std::string> &IncludePaths,
    ArrayRef<std::string> CommandLine,
    clang::ModuleMap *ModuleMap)
  : ModuleMapPath(ModuleMapPath), IncludePaths(IncludePaths),
    CommandLine(CommandLine),
    ModMap(ModuleMap) {}

// Create instance of CoverageChecker, to simplify setting up
// subordinate objects.
std::unique_ptr<CoverageChecker> CoverageChecker::createCoverageChecker(
    StringRef ModuleMapPath, std::vector<std::string> &IncludePaths,
    ArrayRef<std::string> CommandLine, clang::ModuleMap *ModuleMap) {

  return std::make_unique<CoverageChecker>(ModuleMapPath, IncludePaths,
                                            CommandLine, ModuleMap);
}

// Do checks.
// Starting from the directory of the module.modulemap file,
// Find all header files, optionally looking only at files
// covered by the include path options, and compare against
// the headers referenced by the module.modulemap file.
// Display warnings for unaccounted-for header files.
// Returns error_code of 0 if there were no errors or warnings, 1 if there
//   were warnings, 2 if any other problem, such as if a bad
//   module map path argument was specified.
std::error_code CoverageChecker::doChecks() {
  std::error_code returnValue;

  // Collect the headers referenced in the modules.
  collectModuleHeaders();

  // Collect the file system headers.
  if (!collectFileSystemHeaders())
    return std::error_code(2, std::generic_category());

  // Do the checks.  These save the problematic file names.
  findUnaccountedForHeaders();

  // Check for warnings.
  if (!UnaccountedForHeaders.empty())
    returnValue = std::error_code(1, std::generic_category());

  return returnValue;
}

// The following functions are called by doChecks.

// Collect module headers.
// Walks the modules and collects referenced headers into
// ModuleMapHeadersSet.
void CoverageChecker::collectModuleHeaders() {
  for (ModuleMap::module_iterator I = ModMap->module_begin(),
    E = ModMap->module_end();
    I != E; ++I) {
    collectModuleHeaders(*I->second);
  }
}

// Collect referenced headers from one module.
// Collects the headers referenced in the given module into
// ModuleMapHeadersSet.
// FIXME: Doesn't collect files from umbrella header.
bool CoverageChecker::collectModuleHeaders(const Module &Mod) {

  if (std::optional<Module::Header> UmbrellaHeader =
          Mod.getUmbrellaHeaderAsWritten()) {
    // Collect umbrella header.
    ModuleMapHeadersSet.insert(
        ModularizeUtilities::getCanonicalPath(UmbrellaHeader->Entry.getName()));
    // Preprocess umbrella header and collect the headers it references.
    if (!collectUmbrellaHeaderHeaders(UmbrellaHeader->Entry.getName()))
      return false;
  } else if (std::optional<Module::DirectoryName> UmbrellaDir =
                 Mod.getUmbrellaDirAsWritten()) {
    // Collect headers in umbrella directory.
    if (!collectUmbrellaHeaders(UmbrellaDir->Entry.getName()))
      return false;
  }

  for (const auto &Header : Mod.getAllHeaders())
    ModuleMapHeadersSet.insert(
        ModularizeUtilities::getCanonicalPath(Header.Entry.getName()));

  for (auto *Submodule : Mod.submodules())
    collectModuleHeaders(*Submodule);

  return true;
}

// Collect headers from an umbrella directory.
bool CoverageChecker::collectUmbrellaHeaders(StringRef UmbrellaDirName) {
  // Initialize directory name.
  SmallString<256> Directory(ModuleMapDirectory);
  if (UmbrellaDirName.size())
    sys::path::append(Directory, UmbrellaDirName);
  if (Directory.size() == 0)
    Directory = ".";
  // Walk the directory.
  std::error_code EC;
  for (sys::fs::directory_iterator I(Directory.str(), EC), E; I != E;
    I.increment(EC)) {
    if (EC)
      return false;
    std::string File(I->path());
    llvm::ErrorOr<sys::fs::basic_file_status> Status = I->status();
    if (!Status)
      return false;
    sys::fs::file_type Type = Status->type();
    // If the file is a directory, ignore the name and recurse.
    if (Type == sys::fs::file_type::directory_file) {
      if (!collectUmbrellaHeaders(File))
        return false;
      continue;
    }
    // If the file does not have a common header extension, ignore it.
    if (!ModularizeUtilities::isHeader(File))
      continue;
    // Save header name.
    ModuleMapHeadersSet.insert(ModularizeUtilities::getCanonicalPath(File));
  }
  return true;
}

// Collect headers referenced from an umbrella file.
bool
CoverageChecker::collectUmbrellaHeaderHeaders(StringRef UmbrellaHeaderName) {

  SmallString<256> PathBuf(ModuleMapDirectory);

  // If directory is empty, it's the current directory.
  if (ModuleMapDirectory.length() == 0)
    sys::fs::current_path(PathBuf);

  // Create the compilation database.
  std::unique_ptr<CompilationDatabase> Compilations;
  Compilations.reset(new FixedCompilationDatabase(Twine(PathBuf), CommandLine));

  std::vector<std::string> HeaderPath;
  HeaderPath.push_back(std::string(UmbrellaHeaderName));

  // Create the tool and run the compilation.
  ClangTool Tool(*Compilations, HeaderPath);
  int HadErrors = Tool.run(new CoverageCheckerFrontendActionFactory(*this));

  // If we had errors, exit early.
  return !HadErrors;
}

// Called from CoverageCheckerCallbacks to track a header included
// from an umbrella header.
void CoverageChecker::collectUmbrellaHeaderHeader(StringRef HeaderName) {

  SmallString<256> PathBuf(ModuleMapDirectory);
  // If directory is empty, it's the current directory.
  if (ModuleMapDirectory.length() == 0)
    sys::fs::current_path(PathBuf);
  // HeaderName will have an absolute path, so if it's the module map
  // directory, we remove it, also skipping trailing separator.
  if (HeaderName.starts_with(PathBuf))
    HeaderName = HeaderName.substr(PathBuf.size() + 1);
  // Save header name.
  ModuleMapHeadersSet.insert(ModularizeUtilities::getCanonicalPath(HeaderName));
}

// Collect file system header files.
// This function scans the file system for header files,
// starting at the directory of the module.modulemap file,
// optionally filtering out all but the files covered by
// the include path options.
// Returns true if no errors.
bool CoverageChecker::collectFileSystemHeaders() {

  // Get directory containing the module.modulemap file.
  // Might be relative to current directory, absolute, or empty.
  ModuleMapDirectory = ModularizeUtilities::getDirectoryFromPath(ModuleMapPath);

  // If no include paths specified, we do the whole tree starting
  // at the module.modulemap directory.
  if (IncludePaths.size() == 0) {
    if (!collectFileSystemHeaders(StringRef("")))
      return false;
  }
  else {
    // Otherwise we only look at the sub-trees specified by the
    // include paths.
    for (std::vector<std::string>::const_iterator I = IncludePaths.begin(),
      E = IncludePaths.end();
      I != E; ++I) {
      if (!collectFileSystemHeaders(*I))
        return false;
    }
  }

  // Sort it, because different file systems might order the file differently.
  llvm::sort(FileSystemHeaders);

  return true;
}

// Collect file system header files from the given path.
// This function scans the file system for header files,
// starting at the given directory, which is assumed to be
// relative to the directory of the module.modulemap file.
// \returns True if no errors.
bool CoverageChecker::collectFileSystemHeaders(StringRef IncludePath) {

  // Initialize directory name.
  SmallString<256> Directory(ModuleMapDirectory);
  if (IncludePath.size())
    sys::path::append(Directory, IncludePath);
  if (Directory.size() == 0)
    Directory = ".";
  if (IncludePath.starts_with("/") || IncludePath.starts_with("\\") ||
      ((IncludePath.size() >= 2) && (IncludePath[1] == ':'))) {
    llvm::errs() << "error: Include path \"" << IncludePath
      << "\" is not relative to the module map file.\n";
    return false;
  }

  // Recursively walk the directory tree.
  std::error_code EC;
  int Count = 0;
  for (sys::fs::recursive_directory_iterator I(Directory.str(), EC), E; I != E;
    I.increment(EC)) {
    if (EC)
      return false;
    //std::string file(I->path());
    StringRef file(I->path());
    llvm::ErrorOr<sys::fs::basic_file_status> Status = I->status();
    if (!Status)
      return false;
    sys::fs::file_type type = Status->type();
    // If the file is a directory, ignore the name (but still recurses).
    if (type == sys::fs::file_type::directory_file)
      continue;
    // Assume directories or files starting with '.' are private and not to
    // be considered.
    if (file.contains("\\.") || file.contains("/."))
      continue;
    // If the file does not have a common header extension, ignore it.
    if (!ModularizeUtilities::isHeader(file))
      continue;
    // Save header name.
    FileSystemHeaders.push_back(ModularizeUtilities::getCanonicalPath(file));
    Count++;
  }
  if (Count == 0) {
    llvm::errs() << "warning: No headers found in include path: \""
      << IncludePath << "\"\n";
  }
  return true;
}

// Find headers unaccounted-for in module map.
// This function compares the list of collected header files
// against those referenced in the module map.  Display
// warnings for unaccounted-for header files.
// Save unaccounted-for file list for possible.
// fixing action.
// FIXME: There probably needs to be some canonalization
// of file names so that header path can be correctly
// matched.  Also, a map could be used for the headers
// referenced in the module, but
void CoverageChecker::findUnaccountedForHeaders() {
  // Walk over file system headers.
  for (std::vector<std::string>::const_iterator I = FileSystemHeaders.begin(),
    E = FileSystemHeaders.end();
    I != E; ++I) {
    // Look for header in module map.
    if (ModuleMapHeadersSet.insert(*I).second) {
      UnaccountedForHeaders.push_back(*I);
      llvm::errs() << "warning: " << ModuleMapPath
        << " does not account for file: " << *I << "\n";
    }
  }
}
