//===--- CompilationDatabase.cpp - ----------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file contains implementations of the CompilationDatabase base class
//  and the FixedCompilationDatabase.
//
//===----------------------------------------------------------------------===//

#include <sstream>
#include "clang/Tooling/CompilationDatabase.h"
#include "clang/Tooling/CompilationDatabasePluginRegistry.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/system_error.h"

namespace clang {
namespace tooling {

CompilationDatabase::~CompilationDatabase() {}

CompilationDatabase *
CompilationDatabase::loadFromDirectory(StringRef BuildDirectory,
                                       std::string &ErrorMessage) {
  std::stringstream ErrorStream;
  for (CompilationDatabasePluginRegistry::iterator
       It = CompilationDatabasePluginRegistry::begin(),
       Ie = CompilationDatabasePluginRegistry::end();
       It != Ie; ++It) {
    std::string DatabaseErrorMessage;
    OwningPtr<CompilationDatabasePlugin> Plugin(It->instantiate());
    if (CompilationDatabase *DB =
        Plugin->loadFromDirectory(BuildDirectory, DatabaseErrorMessage))
      return DB;
    else
      ErrorStream << It->getName() << ": " << DatabaseErrorMessage << "\n";
  }
  ErrorMessage = ErrorStream.str();
  return NULL;
}

static CompilationDatabase *
findCompilationDatabaseFromDirectory(StringRef Directory,
                                     std::string &ErrorMessage) {
  std::stringstream ErrorStream;
  bool HasErrorMessage = false;
  while (!Directory.empty()) {
    std::string LoadErrorMessage;

    if (CompilationDatabase *DB =
           CompilationDatabase::loadFromDirectory(Directory, LoadErrorMessage))
      return DB;

    if (!HasErrorMessage) {
      ErrorStream << "No compilation database found in " << Directory.str()
                  << " or any parent directory\n" << LoadErrorMessage;
      HasErrorMessage = true;
    }

    Directory = llvm::sys::path::parent_path(Directory);
  }
  ErrorMessage = ErrorStream.str();
  return NULL;
}

CompilationDatabase *
CompilationDatabase::autoDetectFromSource(StringRef SourceFile,
                                          std::string &ErrorMessage) {
  llvm::SmallString<1024> AbsolutePath(getAbsolutePath(SourceFile));
  StringRef Directory = llvm::sys::path::parent_path(AbsolutePath);

  CompilationDatabase *DB = findCompilationDatabaseFromDirectory(Directory,
                                                                 ErrorMessage);

  if (!DB)
    ErrorMessage = ("Could not auto-detect compilation database for file \"" +
                   SourceFile + "\"\n" + ErrorMessage).str();
  return DB;
}

CompilationDatabase *
CompilationDatabase::autoDetectFromDirectory(StringRef SourceDir,
                                             std::string &ErrorMessage) {
  llvm::SmallString<1024> AbsolutePath(getAbsolutePath(SourceDir));

  CompilationDatabase *DB = findCompilationDatabaseFromDirectory(AbsolutePath,
                                                                 ErrorMessage);

  if (!DB)
    ErrorMessage = ("Could not auto-detect compilation database from directory \"" +
                   SourceDir + "\"\n" + ErrorMessage).str();
  return DB;
}

CompilationDatabasePlugin::~CompilationDatabasePlugin() {}

FixedCompilationDatabase *
FixedCompilationDatabase::loadFromCommandLine(int &Argc,
                                              const char **Argv,
                                              Twine Directory) {
  const char **DoubleDash = std::find(Argv, Argv + Argc, StringRef("--"));
  if (DoubleDash == Argv + Argc)
    return NULL;
  std::vector<std::string> CommandLine(DoubleDash + 1, Argv + Argc);
  Argc = DoubleDash - Argv;
  return new FixedCompilationDatabase(Directory, CommandLine);
}

FixedCompilationDatabase::
FixedCompilationDatabase(Twine Directory, ArrayRef<std::string> CommandLine) {
  std::vector<std::string> ToolCommandLine(1, "clang-tool");
  ToolCommandLine.insert(ToolCommandLine.end(),
                         CommandLine.begin(), CommandLine.end());
  CompileCommands.push_back(CompileCommand(Directory, ToolCommandLine));
}

std::vector<CompileCommand>
FixedCompilationDatabase::getCompileCommands(StringRef FilePath) const {
  std::vector<CompileCommand> Result(CompileCommands);
  Result[0].CommandLine.push_back(FilePath);
  return Result;
}

std::vector<std::string>
FixedCompilationDatabase::getAllFiles() const {
  return std::vector<std::string>();
}

// This anchor is used to force the linker to link in the generated object file
// and thus register the JSONCompilationDatabasePlugin.
extern volatile int JSONAnchorSource;
static int JSONAnchorDest = JSONAnchorSource;

} // end namespace tooling
} // end namespace clang
