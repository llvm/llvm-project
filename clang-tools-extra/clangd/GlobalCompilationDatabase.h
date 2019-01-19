//===--- GlobalCompilationDatabase.h -----------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANGD_GLOBALCOMPILATIONDATABASE_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANGD_GLOBALCOMPILATIONDATABASE_H

#include "Function.h"
#include "Path.h"
#include "llvm/ADT/StringMap.h"
#include <memory>
#include <mutex>
#include <vector>

namespace clang {

namespace tooling {
class CompilationDatabase;
struct CompileCommand;
} // namespace tooling

namespace clangd {

class Logger;

struct ProjectInfo {
  // The directory in which the compilation database was discovered.
  // Empty if directory-based compilation database discovery was not used.
  std::string SourceRoot;
};

/// Provides compilation arguments used for parsing C and C++ files.
class GlobalCompilationDatabase {
public:
  virtual ~GlobalCompilationDatabase() = default;

  /// If there are any known-good commands for building this file, returns one.
  /// If the ProjectInfo pointer is set, it will also be populated.
  virtual llvm::Optional<tooling::CompileCommand>
  getCompileCommand(PathRef File, ProjectInfo * = nullptr) const = 0;

  /// Makes a guess at how to build a file.
  /// The default implementation just runs clang on the file.
  /// Clangd should treat the results as unreliable.
  virtual tooling::CompileCommand getFallbackCommand(PathRef File) const;

  using CommandChanged = Event<std::vector<std::string>>;
  /// The callback is notified when files may have new compile commands.
  /// The argument is a list of full file paths.
  CommandChanged::Subscription watch(CommandChanged::Listener L) const {
    return OnCommandChanged.observe(std::move(L));
  }

protected:
  mutable CommandChanged OnCommandChanged;
};

/// Gets compile args from tooling::CompilationDatabases built for parent
/// directories.
class DirectoryBasedGlobalCompilationDatabase
    : public GlobalCompilationDatabase {
public:
  DirectoryBasedGlobalCompilationDatabase(
      llvm::Optional<Path> CompileCommandsDir);
  ~DirectoryBasedGlobalCompilationDatabase() override;

  /// Scans File's parents looking for compilation databases.
  /// Any extra flags will be added.
  llvm::Optional<tooling::CompileCommand>
  getCompileCommand(PathRef File, ProjectInfo * = nullptr) const override;

private:
  tooling::CompilationDatabase *getCDBForFile(PathRef File,
                                              ProjectInfo *) const;
  std::pair<tooling::CompilationDatabase *, /*Cached*/ bool>
  getCDBInDirLocked(PathRef File) const;

  mutable std::mutex Mutex;
  /// Caches compilation databases loaded from directories(keys are
  /// directories).
  mutable llvm::StringMap<std::unique_ptr<clang::tooling::CompilationDatabase>>
      CompilationDatabases;

  /// Used for command argument pointing to folder where compile_commands.json
  /// is located.
  llvm::Optional<Path> CompileCommandsDir;
};

/// Wraps another compilation database, and supports overriding the commands
/// using an in-memory mapping.
class OverlayCDB : public GlobalCompilationDatabase {
public:
  // Base may be null, in which case no entries are inherited.
  // FallbackFlags are added to the fallback compile command.
  OverlayCDB(const GlobalCompilationDatabase *Base,
             std::vector<std::string> FallbackFlags = {});

  llvm::Optional<tooling::CompileCommand>
  getCompileCommand(PathRef File, ProjectInfo * = nullptr) const override;
  tooling::CompileCommand getFallbackCommand(PathRef File) const override;

  /// Sets or clears the compilation command for a particular file.
  void
  setCompileCommand(PathRef File,
                    llvm::Optional<tooling::CompileCommand> CompilationCommand);

private:
  mutable std::mutex Mutex;
  llvm::StringMap<tooling::CompileCommand> Commands; /* GUARDED_BY(Mut) */
  const GlobalCompilationDatabase *Base;
  std::vector<std::string> FallbackFlags;
  CommandChanged::Subscription BaseChanged;
};

} // namespace clangd
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANGD_GLOBALCOMPILATIONDATABASE_H
