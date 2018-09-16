//===--- GlobalCompilationDatabase.cpp ---------------------------*- C++-*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "GlobalCompilationDatabase.h"
#include "Logger.h"
#include "clang/Tooling/CompilationDatabase.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"

namespace clang {
namespace clangd {

tooling::CompileCommand
GlobalCompilationDatabase::getFallbackCommand(PathRef File) const {
  std::vector<std::string> Argv = {"clang"};
  // Clang treats .h files as C by default, resulting in unhelpful diagnostics.
  // Parsing as Objective C++ is friendly to more cases.
  if (llvm::sys::path::extension(File) == ".h")
    Argv.push_back("-xobjective-c++-header");
  Argv.push_back(File);
  return tooling::CompileCommand(llvm::sys::path::parent_path(File),
                                 llvm::sys::path::filename(File),
                                 std::move(Argv),
                                 /*Output=*/"");
}

DirectoryBasedGlobalCompilationDatabase::
    DirectoryBasedGlobalCompilationDatabase(
        llvm::Optional<Path> CompileCommandsDir)
    : CompileCommandsDir(std::move(CompileCommandsDir)) {}

DirectoryBasedGlobalCompilationDatabase::
    ~DirectoryBasedGlobalCompilationDatabase() = default;

llvm::Optional<tooling::CompileCommand>
DirectoryBasedGlobalCompilationDatabase::getCompileCommand(PathRef File) const {
  if (auto CDB = getCDBForFile(File)) {
    auto Candidates = CDB->getCompileCommands(File);
    if (!Candidates.empty()) {
      addExtraFlags(File, Candidates.front());
      return std::move(Candidates.front());
    }
  } else {
    log("Failed to find compilation database for {0}", File);
  }
  return llvm::None;
}

tooling::CompileCommand
DirectoryBasedGlobalCompilationDatabase::getFallbackCommand(
    PathRef File) const {
  auto C = GlobalCompilationDatabase::getFallbackCommand(File);
  addExtraFlags(File, C);
  return C;
}

void DirectoryBasedGlobalCompilationDatabase::setCompileCommandsDir(Path P) {
  std::lock_guard<std::mutex> Lock(Mutex);
  CompileCommandsDir = P;
  CompilationDatabases.clear();
}

void DirectoryBasedGlobalCompilationDatabase::setExtraFlagsForFile(
    PathRef File, std::vector<std::string> ExtraFlags) {
  std::lock_guard<std::mutex> Lock(Mutex);
  ExtraFlagsForFile[File] = std::move(ExtraFlags);
}

void DirectoryBasedGlobalCompilationDatabase::addExtraFlags(
    PathRef File, tooling::CompileCommand &C) const {
  std::lock_guard<std::mutex> Lock(Mutex);

  auto It = ExtraFlagsForFile.find(File);
  if (It == ExtraFlagsForFile.end())
    return;

  auto &Args = C.CommandLine;
  assert(Args.size() >= 2 && "Expected at least [compiler, source file]");
  // The last argument of CommandLine is the name of the input file.
  // Add ExtraFlags before it.
  Args.insert(Args.end() - 1, It->second.begin(), It->second.end());
}

tooling::CompilationDatabase *
DirectoryBasedGlobalCompilationDatabase::getCDBInDirLocked(PathRef Dir) const {
  // FIXME(ibiryukov): Invalidate cached compilation databases on changes
  auto CachedIt = CompilationDatabases.find(Dir);
  if (CachedIt != CompilationDatabases.end())
    return CachedIt->second.get();
  std::string Error = "";
  auto CDB = tooling::CompilationDatabase::loadFromDirectory(Dir, Error);
  auto Result = CDB.get();
  CompilationDatabases.insert(std::make_pair(Dir, std::move(CDB)));
  return Result;
}

tooling::CompilationDatabase *
DirectoryBasedGlobalCompilationDatabase::getCDBForFile(PathRef File) const {
  namespace path = llvm::sys::path;
  assert((path::is_absolute(File, path::Style::posix) ||
          path::is_absolute(File, path::Style::windows)) &&
         "path must be absolute");

  std::lock_guard<std::mutex> Lock(Mutex);
  if (CompileCommandsDir)
    return getCDBInDirLocked(*CompileCommandsDir);
  for (auto Path = path::parent_path(File); !Path.empty();
       Path = path::parent_path(Path))
    if (auto CDB = getCDBInDirLocked(Path))
      return CDB;
  return nullptr;
}

CachingCompilationDb::CachingCompilationDb(
    const GlobalCompilationDatabase &InnerCDB)
    : InnerCDB(InnerCDB) {}

llvm::Optional<tooling::CompileCommand>
CachingCompilationDb::getCompileCommand(PathRef File) const {
  std::unique_lock<std::mutex> Lock(Mut);
  auto It = Cached.find(File);
  if (It != Cached.end())
    return It->second;

  Lock.unlock();
  llvm::Optional<tooling::CompileCommand> Command =
      InnerCDB.getCompileCommand(File);
  Lock.lock();
  return Cached.try_emplace(File, std::move(Command)).first->getValue();
}

tooling::CompileCommand
CachingCompilationDb::getFallbackCommand(PathRef File) const {
  return InnerCDB.getFallbackCommand(File);
}

void CachingCompilationDb::invalidate(PathRef File) {
  std::unique_lock<std::mutex> Lock(Mut);
  Cached.erase(File);
}

void CachingCompilationDb::clear() {
  std::unique_lock<std::mutex> Lock(Mut);
  Cached.clear();
}

llvm::Optional<tooling::CompileCommand>
InMemoryCompilationDb::getCompileCommand(PathRef File) const {
  std::lock_guard<std::mutex> Lock(Mutex);
  auto It = Commands.find(File);
  if (It == Commands.end())
    return None;
  return It->second;
}

bool InMemoryCompilationDb::setCompilationCommandForFile(
    PathRef File, tooling::CompileCommand CompilationCommand) {
  std::unique_lock<std::mutex> Lock(Mutex);
  auto ItInserted = Commands.insert(std::make_pair(File, CompilationCommand));
  if (ItInserted.second)
    return true;
  ItInserted.first->setValue(std::move(CompilationCommand));
  return false;
}

void InMemoryCompilationDb::invalidate(PathRef File) {
  std::unique_lock<std::mutex> Lock(Mutex);
  Commands.erase(File);
}

} // namespace clangd
} // namespace clang
