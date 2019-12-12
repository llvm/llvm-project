//===- ModuleDepCollector.cpp - Callbacks to collect deps -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "clang/Tooling/DependencyScanning/ModuleDepCollector.h"

#include "clang/Frontend/CompilerInstance.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Tooling/DependencyScanning/DependencyScanningWorker.h"

using namespace clang;
using namespace tooling;
using namespace dependencies;

void ModuleDepCollectorPP::FileChanged(SourceLocation Loc,
                                       FileChangeReason Reason,
                                       SrcMgr::CharacteristicKind FileType,
                                       FileID PrevFID) {
  if (Reason != PPCallbacks::EnterFile)
    return;

  SourceManager &SM = Instance.getSourceManager();

  // Dependency generation really does want to go all the way to the
  // file entry for a source location to find out what is depended on.
  // We do not want #line markers to affect dependency generation!
  Optional<FileEntryRef> File =
      SM.getFileEntryRefForID(SM.getFileID(SM.getExpansionLoc(Loc)));
  if (!File)
    return;

  StringRef FileName =
      llvm::sys::path::remove_leading_dotslash(File->getName());

  MDC.MainDeps.push_back(FileName);
}

void ModuleDepCollectorPP::InclusionDirective(
    SourceLocation HashLoc, const Token &IncludeTok, StringRef FileName,
    bool IsAngled, CharSourceRange FilenameRange, const FileEntry *File,
    StringRef SearchPath, StringRef RelativePath, const Module *Imported,
    SrcMgr::CharacteristicKind FileType) {
  if (!File && !Imported) {
    // This is a non-modular include that HeaderSearch failed to find. Add it
    // here as `FileChanged` will never see it.
    MDC.MainDeps.push_back(FileName);
  }

  if (!Imported)
    return;

  MDC.Deps[MDC.ContextHash + Imported->getTopLevelModule()->getFullModuleName()]
      .ImportedByMainFile = true;
  DirectDeps.insert(Imported->getTopLevelModule());
}

void ModuleDepCollectorPP::EndOfMainFile() {
  FileID MainFileID = Instance.getSourceManager().getMainFileID();
  MDC.MainFile =
      Instance.getSourceManager().getFileEntryForID(MainFileID)->getName();

  for (const Module *M : DirectDeps) {
    handleTopLevelModule(M);
  }

  for (auto &&I : MDC.Deps)
    MDC.Consumer.handleModuleDependency(I.second);

  DependencyOutputOptions Opts;
  for (auto &&I : MDC.MainDeps)
    MDC.Consumer.handleFileDependency(Opts, I);
}

void ModuleDepCollectorPP::handleTopLevelModule(const Module *M) {
  assert(M == M->getTopLevelModule() && "Expected top level module!");

  auto ModI = MDC.Deps.insert(
      std::make_pair(MDC.ContextHash + M->getFullModuleName(), ModuleDeps{}));

  if (!ModI.first->second.ModuleName.empty())
    return;

  ModuleDeps &MD = ModI.first->second;

  const FileEntry *ModuleMap = Instance.getPreprocessor()
                                   .getHeaderSearchInfo()
                                   .getModuleMap()
                                   .getContainingModuleMapFile(M);

  MD.ClangModuleMapFile = ModuleMap ? ModuleMap->getName() : "";
  MD.ModuleName = M->getFullModuleName();
  MD.ModulePCMPath = M->getASTFile()->getName();
  MD.ContextHash = MDC.ContextHash;
  serialization::ModuleFile *MF =
      MDC.Instance.getASTReader()->getModuleManager().lookup(M->getASTFile());
  MDC.Instance.getASTReader()->visitInputFiles(
      *MF, true, true, [&](const serialization::InputFile &IF, bool isSystem) {
        MD.FileDeps.insert(IF.getFile()->getName());
      });

  addAllSubmoduleDeps(M, MD);
}

void ModuleDepCollectorPP::addAllSubmoduleDeps(const Module *M,
                                               ModuleDeps &MD) {
  addModuleDep(M, MD);

  for (const Module *SubM : M->submodules())
    addAllSubmoduleDeps(SubM, MD);
}

void ModuleDepCollectorPP::addModuleDep(const Module *M, ModuleDeps &MD) {
  for (const Module *Import : M->Imports) {
    if (Import->getTopLevelModule() != M->getTopLevelModule()) {
      MD.ClangModuleDeps.insert(Import->getTopLevelModuleName());
      handleTopLevelModule(Import->getTopLevelModule());
    }
  }
}

ModuleDepCollector::ModuleDepCollector(CompilerInstance &I,
                                       DependencyConsumer &C)
    : Instance(I), Consumer(C), ContextHash(I.getInvocation().getModuleHash()) {
}

void ModuleDepCollector::attachToPreprocessor(Preprocessor &PP) {
  PP.addPPCallbacks(std::make_unique<ModuleDepCollectorPP>(Instance, *this));
}

void ModuleDepCollector::attachToASTReader(ASTReader &R) {}
