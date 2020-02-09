//===- ModuleDepCollector.h - Callbacks to collect deps ---------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLING_DEPENDENCY_SCANNING_MODULE_DEP_COLLECTOR_H
#define LLVM_CLANG_TOOLING_DEPENDENCY_SCANNING_MODULE_DEP_COLLECTOR_H

#include "clang/Basic/LLVM.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Frontend/Utils.h"
#include "clang/Lex/HeaderSearch.h"
#include "clang/Lex/PPCallbacks.h"
#include "clang/Serialization/ASTReader.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/raw_ostream.h"

#include <string>

namespace clang {
namespace tooling {
namespace dependencies {

class DependencyConsumer;

struct ModuleDeps {
  std::string ModuleName;
  std::string ClangModuleMapFile;
  std::string ModulePCMPath;
  std::string ContextHash;
  llvm::StringSet<> FileDeps;
  llvm::StringSet<> ClangModuleDeps;
  bool ImportedByMainFile = false;
};

class ModuleDepCollector;

class ModuleDepCollectorPP final : public PPCallbacks {
public:
  ModuleDepCollectorPP(CompilerInstance &I, ModuleDepCollector &MDC)
      : Instance(I), MDC(MDC) {}

  void FileChanged(SourceLocation Loc, FileChangeReason Reason,
                   SrcMgr::CharacteristicKind FileType,
                   FileID PrevFID) override;
  void InclusionDirective(SourceLocation HashLoc, const Token &IncludeTok,
                          StringRef FileName, bool IsAngled,
                          CharSourceRange FilenameRange, const FileEntry *File,
                          StringRef SearchPath, StringRef RelativePath,
                          const Module *Imported,
                          SrcMgr::CharacteristicKind FileType) override;

  void EndOfMainFile() override;

private:
  CompilerInstance &Instance;
  ModuleDepCollector &MDC;
  llvm::DenseSet<const Module *> DirectDeps;

  void handleTopLevelModule(const Module *M);
  void addAllSubmoduleDeps(const Module *M, ModuleDeps &MD);
  void addModuleDep(const Module *M, ModuleDeps &MD);

  void addDirectDependencies(const Module *Mod);
};

class ModuleDepCollector final : public DependencyCollector {
public:
  ModuleDepCollector(CompilerInstance &I, DependencyConsumer &C);

  void attachToPreprocessor(Preprocessor &PP) override;
  void attachToASTReader(ASTReader &R) override;

private:
  friend ModuleDepCollectorPP;

  CompilerInstance &Instance;
  DependencyConsumer &Consumer;
  std::string MainFile;
  std::string ContextHash;
  std::vector<std::string> MainDeps;
  std::unordered_map<std::string, ModuleDeps> Deps;
};

} // end namespace dependencies
} // end namespace tooling
} // end namespace clang

#endif // LLVM_CLANG_TOOLING_DEPENDENCY_SCANNING_MODULE_DEP_COLLECTOR_H
