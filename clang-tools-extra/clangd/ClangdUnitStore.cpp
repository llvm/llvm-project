//===--- ClangdUnitStore.cpp - A ClangdUnits container -----------*-C++-*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "ClangdUnitStore.h"
#include "llvm/Support/Path.h"
#include <algorithm>

using namespace clang::clangd;
using namespace clang;

std::shared_ptr<CppFile> CppFileCollection::removeIfPresent(PathRef File) {
  std::lock_guard<std::mutex> Lock(Mutex);

  auto It = OpenedFiles.find(File);
  if (It == OpenedFiles.end())
    return nullptr;

  std::shared_ptr<CppFile> Result = It->second;
  OpenedFiles.erase(It);
  return Result;
}

CppFileCollection::RecreateResult
CppFileCollection::recreateFileIfCompileCommandChanged(
    PathRef File, PathRef ResourceDir, GlobalCompilationDatabase &CDB,
    bool StorePreamblesInMemory, std::shared_ptr<PCHContainerOperations> PCHs) {
  auto NewCommand = getCompileCommand(CDB, File, ResourceDir);

  std::lock_guard<std::mutex> Lock(Mutex);

  RecreateResult Result;

  auto It = OpenedFiles.find(File);
  if (It == OpenedFiles.end()) {
    It = OpenedFiles
             .try_emplace(File, CppFile::Create(File, std::move(NewCommand),
                                                StorePreamblesInMemory,
                                                std::move(PCHs), ASTCallback))
             .first;
  } else if (!compileCommandsAreEqual(It->second->getCompileCommand(),
                                      NewCommand)) {
    Result.RemovedFile = std::move(It->second);
    It->second =
        CppFile::Create(File, std::move(NewCommand), StorePreamblesInMemory,
                        std::move(PCHs), ASTCallback);
  }
  Result.FileInCollection = It->second;
  return Result;
}

tooling::CompileCommand
CppFileCollection::getCompileCommand(GlobalCompilationDatabase &CDB,
                                     PathRef File, PathRef ResourceDir) {
  llvm::Optional<tooling::CompileCommand> C = CDB.getCompileCommand(File);
  if (!C) // FIXME: Suppress diagnostics? Let the user know?
    C = CDB.getFallbackCommand(File);

  // Inject the resource dir.
  // FIXME: Don't overwrite it if it's already there.
  C->CommandLine.push_back("-resource-dir=" + ResourceDir.str());
  return std::move(*C);
}

bool CppFileCollection::compileCommandsAreEqual(
    tooling::CompileCommand const &LHS, tooling::CompileCommand const &RHS) {
  // tooling::CompileCommand.Output is ignored, it's not relevant for clangd.
  return LHS.Directory == RHS.Directory &&
         LHS.CommandLine.size() == RHS.CommandLine.size() &&
         std::equal(LHS.CommandLine.begin(), LHS.CommandLine.end(),
                    RHS.CommandLine.begin());
}
