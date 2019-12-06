//===--- GlobalCompilationDatabase.h -----------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANGD_GLOBALCOMPILATIONDATABASE_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANGD_GLOBALCOMPILATIONDATABASE_H

#include "CompileCommands.h"
#include "Function.h"
#include "Path.h"
#include "clang/Tooling/ArgumentsAdjusters.h"
#include "clang/Tooling/CompilationDatabase.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/StringMap.h"
#include <memory>
#include <mutex>
#include <vector>

namespace clang {
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
  virtual llvm::Optional<tooling::CompileCommand>
  getCompileCommand(PathRef File) const = 0;

  /// Finds the closest project to \p File.
  virtual llvm::Optional<ProjectInfo> getProjectInfo(PathRef File) const {
    return llvm::None;
  }

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
  /// Might trigger OnCommandChanged, if CDB wasn't broadcasted yet.
  llvm::Optional<tooling::CompileCommand>
  getCompileCommand(PathRef File) const override;

  /// Returns the path to first directory containing a compilation database in
  /// \p File's parents.
  llvm::Optional<ProjectInfo> getProjectInfo(PathRef File) const override;

private:
  /// Caches compilation databases loaded from directories.
  struct CachedCDB {
    std::string Path; // Not case-folded.
    std::unique_ptr<clang::tooling::CompilationDatabase> CDB = nullptr;
    bool SentBroadcast = false;
  };
  CachedCDB &getCDBInDirLocked(PathRef File) const;

  struct CDBLookupRequest {
    PathRef FileName;
    // Whether this lookup should trigger discovery of the CDB found.
    bool ShouldBroadcast = false;
  };
  struct CDBLookupResult {
    tooling::CompilationDatabase *CDB = nullptr;
    ProjectInfo PI;
  };
  llvm::Optional<CDBLookupResult> lookupCDB(CDBLookupRequest Request) const;

  // Performs broadcast on governed files.
  void broadcastCDB(CDBLookupResult Res) const;

  mutable std::mutex Mutex;
  // Keyed by possibly-case-folded directory path.
  mutable llvm::StringMap<CachedCDB> CompilationDatabases;

  /// Used for command argument pointing to folder where compile_commands.json
  /// is located.
  llvm::Optional<Path> CompileCommandsDir;
};

/// Extracts system include search path from drivers matching QueryDriverGlobs
/// and adds them to the compile flags. Base may not be nullptr.
/// Returns Base when \p QueryDriverGlobs is empty.
std::unique_ptr<GlobalCompilationDatabase>
getQueryDriverDatabase(llvm::ArrayRef<std::string> QueryDriverGlobs,
                       std::unique_ptr<GlobalCompilationDatabase> Base);


/// Wraps another compilation database, and supports overriding the commands
/// using an in-memory mapping.
class OverlayCDB : public GlobalCompilationDatabase {
public:
  // Base may be null, in which case no entries are inherited.
  // FallbackFlags are added to the fallback compile command.
  // Adjuster is applied to all commands, fallback or not.
  OverlayCDB(const GlobalCompilationDatabase *Base,
             std::vector<std::string> FallbackFlags = {},
             tooling::ArgumentsAdjuster Adjuster = nullptr);

  llvm::Optional<tooling::CompileCommand>
  getCompileCommand(PathRef File) const override;
  tooling::CompileCommand getFallbackCommand(PathRef File) const override;
  llvm::Optional<ProjectInfo> getProjectInfo(PathRef File) const override;

  /// Sets or clears the compilation command for a particular file.
  void
  setCompileCommand(PathRef File,
                    llvm::Optional<tooling::CompileCommand> CompilationCommand);

private:
  mutable std::mutex Mutex;
  llvm::StringMap<tooling::CompileCommand> Commands; /* GUARDED_BY(Mut) */
  const GlobalCompilationDatabase *Base;
  tooling::ArgumentsAdjuster ArgsAdjuster;
  std::vector<std::string> FallbackFlags;
  CommandChanged::Subscription BaseChanged;
};

} // namespace clangd
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANGD_GLOBALCOMPILATIONDATABASE_H
