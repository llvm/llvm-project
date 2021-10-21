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

static void optimizeHeaderSearchOpts(HeaderSearchOptions &Opts,
                                     ASTReader &Reader,
                                     const serialization::ModuleFile &MF) {
  // Only preserve search paths that were used during the dependency scan.
  std::vector<HeaderSearchOptions::Entry> Entries = Opts.UserEntries;
  Opts.UserEntries.clear();
  for (unsigned I = 0; I < Entries.size(); ++I)
    if (MF.SearchPathUsage[I])
      Opts.UserEntries.push_back(Entries[I]);
}

CompilerInvocation ModuleDepCollector::makeInvocationForModuleBuildWithoutPaths(
    const ModuleDeps &Deps,
    llvm::function_ref<void(CompilerInvocation &)> Optimize) const {
  // Make a deep copy of the original Clang invocation.
  CompilerInvocation CI(OriginalInvocation);

  CI.getLangOpts()->resetNonModularOptions();
  CI.getPreprocessorOpts().resetNonModularOptions();

  // Remove options incompatible with explicit module build.
  CI.getFrontendOpts().Inputs.clear();
  CI.getFrontendOpts().OutputFile.clear();

  CI.getFrontendOpts().ProgramAction = frontend::GenerateModule;
  CI.getLangOpts()->ModuleName = Deps.ID.ModuleName;
  CI.getFrontendOpts().IsSystemModule = Deps.IsSystem;

  CI.getLangOpts()->ImplicitModules = false;

  // Report the prebuilt modules this module uses.
  for (const auto &PrebuiltModule : Deps.PrebuiltModuleDeps) {
    CI.getFrontendOpts().ModuleFiles.push_back(PrebuiltModule.PCMFile);
    CI.getFrontendOpts().ModuleMapFiles.push_back(PrebuiltModule.ModuleMapFile);
  }

  Optimize(CI);

  // The original invocation probably didn't have strict context hash enabled.
  // We will use the context hash of this invocation to distinguish between
  // multiple incompatible versions of the same module and will use it when
  // reporting dependencies to the clients. Let's make sure we're using
  // **strict** context hash in order to prevent accidental sharing of
  // incompatible modules (e.g. with differences in search paths).
  CI.getHeaderSearchOpts().ModulesStrictContextHash = true;

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

std::vector<std::string>
ModuleDeps::getAdditionalArgsWithoutModulePaths() const {
  std::vector<std::string> Ret{
    "-remove-preceeding-explicit-module-build-incompatible-options",
    "-fno-implicit-modules", "-emit-module", "-fmodule-name=" + ID.ModuleName,
  };

  if (IsSystem)
    Ret.push_back("-fsystem-module");

  if (Invocation.getLangOpts()->NeededByPCHOrCompilationUsesPCH)
    Ret.push_back("-fmodule-related-to-pch");

  return Ret;
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
    MDC.ContextHash = MDC.ScanInstance.getInvocation().getModuleHash(
        MDC.ScanInstance.getDiagnostics());
    MDC.Consumer.handleContextHash(MDC.ContextHash);
  }

  SourceManager &SM = MDC.ScanInstance.getSourceManager();

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

  if (MDC.isPrebuiltModule(TopLevelModule))
    DirectPrebuiltModularDeps.insert(TopLevelModule);
  else
    DirectModularDeps.insert(TopLevelModule);
}

void ModuleDepCollectorPP::EndOfMainFile() {
  FileID MainFileID = MDC.ScanInstance.getSourceManager().getMainFileID();
  MDC.MainFile = std::string(MDC.ScanInstance.getSourceManager()
                                 .getFileEntryForID(MainFileID)
                                 ->getName());

  if (!MDC.ScanInstance.getPreprocessorOpts().ImplicitPCHInclude.empty())
    MDC.FileDeps.push_back(
        MDC.ScanInstance.getPreprocessorOpts().ImplicitPCHInclude);

  for (const Module *M : DirectModularDeps) {
    // A top-level module might not be actually imported as a module when
    // -fmodule-name is used to compile a translation unit that imports this
    // module. In that case it can be skipped. The appropriate header
    // dependencies will still be reported as expected.
    if (!M->getASTFile())
      continue;
    handleTopLevelModule(M);
  }

  MDC.Consumer.handleDependencyOutputOpts(*MDC.Opts);

  for (auto &&I : MDC.ModularDeps)
    MDC.Consumer.handleModuleDependency(I.second);

  for (auto &&I : MDC.FileDeps)
    MDC.Consumer.handleFileDependency(I);

  for (auto &&I : DirectPrebuiltModularDeps)
    MDC.Consumer.handlePrebuiltModuleDependency(PrebuiltModuleDep{I});
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

  const FileEntry *ModuleMap = MDC.ScanInstance.getPreprocessor()
                                   .getHeaderSearchInfo()
                                   .getModuleMap()
                                   .getModuleMapFileForUniquing(M);
  MD.ClangModuleMapFile = std::string(ModuleMap ? ModuleMap->getName() : "");

  serialization::ModuleFile *MF =
      MDC.ScanInstance.getASTReader()->getModuleManager().lookup(
          M->getASTFile());
  MDC.ScanInstance.getASTReader()->visitInputFiles(
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

  // Add direct prebuilt module dependencies now, so that we can use them when
  // creating a CompilerInvocation and computing context hash for this
  // ModuleDeps instance.
  llvm::DenseSet<const Module *> SeenModules;
  addAllSubmodulePrebuiltDeps(M, MD, SeenModules);

  MD.Invocation = MDC.makeInvocationForModuleBuildWithoutPaths(
      MD, [&](CompilerInvocation &CI) {
        if (MDC.OptimizeArgs)
          optimizeHeaderSearchOpts(CI.getHeaderSearchOpts(),
                                   *MDC.ScanInstance.getASTReader(), *MF);
      });
  MD.ID.ContextHash = MD.Invocation.getModuleHash(Instance.getDiagnostics());

  llvm::DenseSet<const Module *> AddedModules;
  addAllSubmoduleDeps(M, MD, AddedModules);

  return MD.ID;
}

void ModuleDepCollectorPP::addAllSubmodulePrebuiltDeps(
    const Module *M, ModuleDeps &MD,
    llvm::DenseSet<const Module *> &SeenSubmodules) {
  addModulePrebuiltDeps(M, MD, SeenSubmodules);

  for (const Module *SubM : M->submodules())
    addAllSubmodulePrebuiltDeps(SubM, MD, SeenSubmodules);
}

void ModuleDepCollectorPP::addModulePrebuiltDeps(
    const Module *M, ModuleDeps &MD,
    llvm::DenseSet<const Module *> &SeenSubmodules) {
  for (const Module *Import : M->Imports)
    if (Import->getTopLevelModule() != M->getTopLevelModule())
      if (MDC.isPrebuiltModule(Import->getTopLevelModule()))
        if (SeenSubmodules.insert(Import->getTopLevelModule()).second)
          MD.PrebuiltModuleDeps.emplace_back(Import->getTopLevelModule());
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
    if (Import->getTopLevelModule() != M->getTopLevelModule() &&
        !MDC.isPrebuiltModule(Import)) {
      ModuleID ImportID = handleTopLevelModule(Import->getTopLevelModule());
      if (AddedModules.insert(Import->getTopLevelModule()).second)
        MD.ClangModuleDeps.push_back(ImportID);
    }
  }
}

ModuleDepCollector::ModuleDepCollector(
    std::unique_ptr<DependencyOutputOptions> Opts,
    CompilerInstance &ScanInstance, DependencyConsumer &C,
    CompilerInvocation &&OriginalCI, bool OptimizeArgs)
    : ScanInstance(ScanInstance), Consumer(C), Opts(std::move(Opts)),
      OriginalInvocation(std::move(OriginalCI)), OptimizeArgs(OptimizeArgs) {}

void ModuleDepCollector::attachToPreprocessor(Preprocessor &PP) {
  PP.addPPCallbacks(std::make_unique<ModuleDepCollectorPP>(*this));
}

void ModuleDepCollector::attachToASTReader(ASTReader &R) {}

bool ModuleDepCollector::isPrebuiltModule(const Module *M) {
  std::string Name(M->getTopLevelModuleName());
  const auto &PrebuiltModuleFiles =
      ScanInstance.getHeaderSearchOpts().PrebuiltModuleFiles;
  auto PrebuiltModuleFileIt = PrebuiltModuleFiles.find(Name);
  if (PrebuiltModuleFileIt == PrebuiltModuleFiles.end())
    return false;
  assert("Prebuilt module came from the expected AST file" &&
         PrebuiltModuleFileIt->second == M->getASTFile()->getName());
  return true;
}
