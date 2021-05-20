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
#include "llvm/Support/StringSaver.h"

using namespace clang;
using namespace tooling;
using namespace dependencies;

static CompilerInvocation
makeInvocationForModuleBuildWithoutPaths(const ModuleDeps &Deps,
                                         const CompilerInvocation &Invocation) {
  // Make a deep copy of the invocation.
  CompilerInvocation CI(Invocation);

  // Remove options incompatible with explicit module build.
  CI.getFrontendOpts().Inputs.clear();
  CI.getFrontendOpts().OutputFile.clear();

  CI.getFrontendOpts().ProgramAction = frontend::GenerateModule;
  CI.getLangOpts()->ModuleName = Deps.ID.ModuleName;
  CI.getFrontendOpts().IsSystemModule = Deps.IsSystem;

  CI.getLangOpts()->ImplicitModules = false;

  return CI;
}

static std::vector<std::string>
serializeCompilerInvocation(const CompilerInvocation &CI) {
  // Set up string allocator.
  llvm::BumpPtrAllocator Alloc;
  llvm::StringSaver Strings(Alloc);
  auto SA = [&Strings](const Twine &Arg) { return Strings.save(Arg).data(); };

  // Synthesize full command line from the CompilerInvocation, including "-cc1".
  SmallVector<const char *, 32> Args{"-cc1"};
  CI.generateCC1CommandLine(Args, SA);

  // Convert arguments to the return type.
  return std::vector<std::string>{Args.begin(), Args.end()};
}

std::vector<std::string> ModuleDeps::getCanonicalCommandLine(
    std::function<StringRef(ModuleID)> LookupPCMPath,
    std::function<const ModuleDeps &(ModuleID)> LookupModuleDeps) const {
  CompilerInvocation CI(Invocation);
  FrontendOptions &FrontendOpts = CI.getFrontendOpts();

  InputKind ModuleMapInputKind(FrontendOpts.DashX.getLanguage(),
                               InputKind::Format::ModuleMap);
  FrontendOpts.Inputs.emplace_back(ClangModuleMapFile, ModuleMapInputKind);
  FrontendOpts.OutputFile = std::string(LookupPCMPath(ID));

  dependencies::detail::collectPCMAndModuleMapPaths(
      ClangModuleDeps, LookupPCMPath, LookupModuleDeps,
      FrontendOpts.ModuleFiles, FrontendOpts.ModuleMapFiles);

  return serializeCompilerInvocation(CI);
}

std::vector<std::string>
ModuleDeps::getCanonicalCommandLineWithoutModulePaths() const {
  return serializeCompilerInvocation(Invocation);
}

void dependencies::detail::collectPCMAndModuleMapPaths(
    llvm::ArrayRef<ModuleID> Modules,
    std::function<StringRef(ModuleID)> LookupPCMPath,
    std::function<const ModuleDeps &(ModuleID)> LookupModuleDeps,
    std::vector<std::string> &PCMPaths, std::vector<std::string> &ModMapPaths) {
  llvm::StringSet<> AlreadyAdded;

  std::function<void(llvm::ArrayRef<ModuleID>)> AddArgs =
      [&](llvm::ArrayRef<ModuleID> Modules) {
        for (const ModuleID &MID : Modules) {
          if (!AlreadyAdded.insert(MID.ModuleName + MID.ContextHash).second)
            continue;
          const ModuleDeps &M = LookupModuleDeps(MID);
          // Depth first traversal.
          AddArgs(M.ClangModuleDeps);
          PCMPaths.push_back(LookupPCMPath(MID).str());
          if (!M.ClangModuleMapFile.empty())
            ModMapPaths.push_back(M.ClangModuleMapFile);
        }
      };

  AddArgs(Modules);
}

void ModuleDepCollectorPP::FileChanged(SourceLocation Loc,
                                       FileChangeReason Reason,
                                       SrcMgr::CharacteristicKind FileType,
                                       FileID PrevFID) {
  if (Reason != PPCallbacks::EnterFile)
    return;
  
  // This has to be delayed as the context hash can change at the start of
  // `CompilerInstance::ExecuteAction`.
  if (MDC.ContextHash.empty()) {
    MDC.ContextHash = Instance.getInvocation().getModuleHash();
    MDC.Consumer.handleContextHash(MDC.ContextHash);
  }

  SourceManager &SM = Instance.getSourceManager();

  // Dependency generation really does want to go all the way to the
  // file entry for a source location to find out what is depended on.
  // We do not want #line markers to affect dependency generation!
  if (Optional<StringRef> Filename =
          SM.getNonBuiltinFilenameForID(SM.getFileID(SM.getExpansionLoc(Loc))))
    MDC.FileDeps.push_back(
        std::string(llvm::sys::path::remove_leading_dotslash(*Filename)));
}

void ModuleDepCollectorPP::InclusionDirective(
    SourceLocation HashLoc, const Token &IncludeTok, StringRef FileName,
    bool IsAngled, CharSourceRange FilenameRange, const FileEntry *File,
    StringRef SearchPath, StringRef RelativePath, const Module *Imported,
    SrcMgr::CharacteristicKind FileType) {
  if (!File && !Imported) {
    // This is a non-modular include that HeaderSearch failed to find. Add it
    // here as `FileChanged` will never see it.
    MDC.FileDeps.push_back(std::string(FileName));
  }
  handleImport(Imported);
}

void ModuleDepCollectorPP::moduleImport(SourceLocation ImportLoc,
                                        ModuleIdPath Path,
                                        const Module *Imported) {
  handleImport(Imported);
}

void ModuleDepCollectorPP::handleImport(const Module *Imported) {
  if (!Imported)
    return;

  const Module *TopLevelModule = Imported->getTopLevelModule();
  DirectModularDeps.insert(TopLevelModule);
}

void ModuleDepCollectorPP::EndOfMainFile() {
  FileID MainFileID = Instance.getSourceManager().getMainFileID();
  MDC.MainFile = std::string(
      Instance.getSourceManager().getFileEntryForID(MainFileID)->getName());

  for (const Module *M : DirectModularDeps)
    handleTopLevelModule(M);

  for (auto &&I : MDC.ModularDeps)
    MDC.Consumer.handleModuleDependency(I.second);

  for (auto &&I : MDC.FileDeps)
    MDC.Consumer.handleFileDependency(*MDC.Opts, I);
}

ModuleID ModuleDepCollectorPP::handleTopLevelModule(const Module *M) {
  assert(M == M->getTopLevelModule() && "Expected top level module!");

  // If this module has been handled already, just return its ID.
  auto ModI = MDC.ModularDeps.insert({M, ModuleDeps{}});
  if (!ModI.second)
    return ModI.first->second.ID;

  ModuleDeps &MD = ModI.first->second;

  MD.ID.ModuleName = M->getFullModuleName();
  MD.ImportedByMainFile = DirectModularDeps.contains(M);
  MD.ImplicitModulePCMPath = std::string(M->getASTFile()->getName());
  MD.IsSystem = M->IsSystem;

  const FileEntry *ModuleMap = Instance.getPreprocessor()
                                   .getHeaderSearchInfo()
                                   .getModuleMap()
                                   .getModuleMapFileForUniquing(M);
  MD.ClangModuleMapFile = std::string(ModuleMap ? ModuleMap->getName() : "");

  serialization::ModuleFile *MF =
      MDC.Instance.getASTReader()->getModuleManager().lookup(M->getASTFile());
  MDC.Instance.getASTReader()->visitInputFiles(
      *MF, true, true, [&](const serialization::InputFile &IF, bool isSystem) {
        // __inferred_module.map is the result of the way in which an implicit
        // module build handles inferred modules. It adds an overlay VFS with
        // this file in the proper directory and relies on the rest of Clang to
        // handle it like normal. With explicitly built modules we don't need
        // to play VFS tricks, so replace it with the correct module map.
        if (IF.getFile()->getName().endswith("__inferred_module.map")) {
          MD.FileDeps.insert(ModuleMap->getName());
          return;
        }
        MD.FileDeps.insert(IF.getFile()->getName());
      });

  MD.Invocation =
      makeInvocationForModuleBuildWithoutPaths(MD, Instance.getInvocation());
  MD.ID.ContextHash = MD.Invocation.getModuleHash();

  llvm::DenseSet<const Module *> AddedModules;
  addAllSubmoduleDeps(M, MD, AddedModules);

  return MD.ID;
}

void ModuleDepCollectorPP::addAllSubmoduleDeps(
    const Module *M, ModuleDeps &MD,
    llvm::DenseSet<const Module *> &AddedModules) {
  addModuleDep(M, MD, AddedModules);

  for (const Module *SubM : M->submodules())
    addAllSubmoduleDeps(SubM, MD, AddedModules);
}

void ModuleDepCollectorPP::addModuleDep(
    const Module *M, ModuleDeps &MD,
    llvm::DenseSet<const Module *> &AddedModules) {
  for (const Module *Import : M->Imports) {
    if (Import->getTopLevelModule() != M->getTopLevelModule()) {
      ModuleID ImportID = handleTopLevelModule(Import->getTopLevelModule());
      if (AddedModules.insert(Import->getTopLevelModule()).second)
        MD.ClangModuleDeps.push_back(ImportID);
    }
  }
}

ModuleDepCollector::ModuleDepCollector(
    std::unique_ptr<DependencyOutputOptions> Opts, CompilerInstance &I,
    DependencyConsumer &C)
    : Instance(I), Consumer(C), Opts(std::move(Opts)) {}

void ModuleDepCollector::attachToPreprocessor(Preprocessor &PP) {
  PP.addPPCallbacks(std::make_unique<ModuleDepCollectorPP>(Instance, *this));
}

void ModuleDepCollector::attachToASTReader(ASTReader &R) {}
