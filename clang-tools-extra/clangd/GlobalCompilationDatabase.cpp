//===--- GlobalCompilationDatabase.cpp --------------------------*- C++-*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===---------------------------------------------------------------------===//

#include "GlobalCompilationDatabase.h"
#include "clang/Tooling/CompilationDatabase.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "Logger.h"

namespace clang {
namespace clangd {

static void addExtraFlags(tooling::CompileCommand &Command,
                          const std::vector<std::string> &ExtraFlags) {
  if (ExtraFlags.empty())
    return;
  assert(Command.CommandLine.size() >= 2 &&
         "Expected a command line containing at least 2 arguments, the "
         "compiler binary and the output file");
  // The last argument of CommandLine is the name of the input file.
  // Add ExtraFlags before it.
  auto It = Command.CommandLine.end();
  --It;
  Command.CommandLine.insert(It, ExtraFlags.begin(), ExtraFlags.end());
}

tooling::CompileCommand getDefaultCompileCommand(PathRef File) {
  std::vector<std::string> CommandLine{"clang", "-fsyntax-only", File.str()};
  return tooling::CompileCommand(llvm::sys::path::parent_path(File),
                                 llvm::sys::path::filename(File), CommandLine,
                                 /*Output=*/"");
}

DirectoryBasedGlobalCompilationDatabase::
    DirectoryBasedGlobalCompilationDatabase(clangd::Logger &Logger)
    : Logger(Logger) {}

std::vector<tooling::CompileCommand>
DirectoryBasedGlobalCompilationDatabase::getCompileCommands(PathRef File) {
  std::vector<tooling::CompileCommand> Commands;

  auto CDB = getCompilationDatabase(File);
  if (CDB)
    Commands = CDB->getCompileCommands(File);
  if (Commands.empty())
    Commands.push_back(getDefaultCompileCommand(File));

  auto It = ExtraFlagsForFile.find(File);
  if (It != ExtraFlagsForFile.end()) {
    // Append the user-specified flags to the compile commands.
    for (tooling::CompileCommand &Command : Commands)
      addExtraFlags(Command, It->second);
  }

  return Commands;
}

void DirectoryBasedGlobalCompilationDatabase::setExtraFlagsForFile(
    PathRef File, std::vector<std::string> ExtraFlags) {
  ExtraFlagsForFile[File] = std::move(ExtraFlags);
}

tooling::CompilationDatabase *
DirectoryBasedGlobalCompilationDatabase::getCompilationDatabase(PathRef File) {
  std::lock_guard<std::mutex> Lock(Mutex);

  namespace path = llvm::sys::path;

  assert((path::is_absolute(File, path::Style::posix) ||
          path::is_absolute(File, path::Style::windows)) &&
         "path must be absolute");

  for (auto Path = path::parent_path(File); !Path.empty();
       Path = path::parent_path(Path)) {

    auto CachedIt = CompilationDatabases.find(Path);
    if (CachedIt != CompilationDatabases.end())
      return CachedIt->second.get();

    std::string Error;
    auto CDB = tooling::CompilationDatabase::loadFromDirectory(Path, Error);
    if (!CDB)
      continue;

    // FIXME(ibiryukov): Invalidate cached compilation databases on changes
    auto Result = CDB.get();
    CompilationDatabases.insert(std::make_pair(Path, std::move(CDB)));
    return Result;
  }

  Logger.log("Failed to find compilation database for " + Twine(File) + "\n");
  return nullptr;
}

} // namespace clangd
} // namespace clang
