//===- ModuleDepCollector.cpp - Callbacks to collect deps -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Tooling/DependencyScanning/ModuleDepCollector.h"

#include "clang/Basic/MakeSupport.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Tooling/DependencyScanning/DependencyScanningWorker.h"
#include "llvm/Support/BLAKE3.h"
#include "llvm/Support/StringSaver.h"
#include <optional>

using namespace clang;
using namespace tooling;
using namespace dependencies;

static void optimizeHeaderSearchOpts(HeaderSearchOptions &Opts,
                                     ASTReader &Reader,
                                     const serialization::ModuleFile &MF) {
  // Only preserve search paths that were used during the dependency scan.
  std::vector<HeaderSearchOptions::Entry> Entries = Opts.UserEntries;
  Opts.UserEntries.clear();

  llvm::BitVector SearchPathUsage(Entries.size());
  llvm::DenseSet<const serialization::ModuleFile *> Visited;
  std::function<void(const serialization::ModuleFile *)> VisitMF =
      [&](const serialization::ModuleFile *MF) {
        SearchPathUsage |= MF->SearchPathUsage;
        Visited.insert(MF);
        for (const serialization::ModuleFile *Import : MF->Imports)
          if (!Visited.contains(Import))
            VisitMF(Import);
      };
  VisitMF(&MF);

  for (auto Idx : SearchPathUsage.set_bits())
    Opts.UserEntries.push_back(Entries[Idx]);
}

static std::vector<std::string> splitString(std::string S, char Separator) {
  SmallVector<StringRef> Segments;
  StringRef(S).split(Segments, Separator, /*MaxSplit=*/-1, /*KeepEmpty=*/false);
  std::vector<std::string> Result;
  Result.reserve(Segments.size());
  for (StringRef Segment : Segments)
    Result.push_back(Segment.str());
  return Result;
}

void ModuleDepCollector::addOutputPaths(CompilerInvocation &CI,
                                        ModuleDeps &Deps) {
  CI.getFrontendOpts().OutputFile =
      Consumer.lookupModuleOutput(Deps.ID, ModuleOutputKind::ModuleFile);
  if (!CI.getDiagnosticOpts().DiagnosticSerializationFile.empty())
    CI.getDiagnosticOpts().DiagnosticSerializationFile =
        Consumer.lookupModuleOutput(
            Deps.ID, ModuleOutputKind::DiagnosticSerializationFile);
  if (!CI.getDependencyOutputOpts().OutputFile.empty()) {
    CI.getDependencyOutputOpts().OutputFile =
        Consumer.lookupModuleOutput(Deps.ID, ModuleOutputKind::DependencyFile);
    CI.getDependencyOutputOpts().Targets =
        splitString(Consumer.lookupModuleOutput(
                        Deps.ID, ModuleOutputKind::DependencyTargets),
                    '\0');
    if (!CI.getDependencyOutputOpts().OutputFile.empty() &&
        CI.getDependencyOutputOpts().Targets.empty()) {
      // Fallback to -o as dependency target, as in the driver.
      SmallString<128> Target;
      quoteMakeTarget(CI.getFrontendOpts().OutputFile, Target);
      CI.getDependencyOutputOpts().Targets.push_back(std::string(Target));
    }
  }
}

CompilerInvocation
ModuleDepCollector::makeInvocationForModuleBuildWithoutOutputs(
    const ModuleDeps &Deps,
    llvm::function_ref<void(CompilerInvocation &)> Optimize) const {
  // Make a deep copy of the original Clang invocation.
  CompilerInvocation CI(OriginalInvocation);

  CI.resetNonModularOptions();
  CI.clearImplicitModuleBuildOptions();

  // Remove options incompatible with explicit module build or are likely to
  // differ between identical modules discovered from different translation
  // units.
  CI.getFrontendOpts().Inputs.clear();
  CI.getFrontendOpts().OutputFile.clear();

  // TODO: Figure out better way to set options to their default value.
  CI.getCodeGenOpts().MainFileName.clear();
  CI.getCodeGenOpts().DwarfDebugFlags.clear();
  if (!CI.getLangOpts()->ModulesCodegen) {
    CI.getCodeGenOpts().DebugCompilationDir.clear();
    CI.getCodeGenOpts().CoverageCompilationDir.clear();
  }

  // Map output paths that affect behaviour to "-" so their existence is in the
  // context hash. The final path will be computed in addOutputPaths.
  if (!CI.getDiagnosticOpts().DiagnosticSerializationFile.empty())
    CI.getDiagnosticOpts().DiagnosticSerializationFile = "-";
  if (!CI.getDependencyOutputOpts().OutputFile.empty())
    CI.getDependencyOutputOpts().OutputFile = "-";
  CI.getDependencyOutputOpts().Targets.clear();

  CI.getFrontendOpts().ProgramAction = frontend::GenerateModule;
  CI.getLangOpts()->ModuleName = Deps.ID.ModuleName;
  CI.getFrontendOpts().IsSystemModule = Deps.IsSystem;

  // Inputs
  InputKind ModuleMapInputKind(CI.getFrontendOpts().DashX.getLanguage(),
                               InputKind::Format::ModuleMap);
  CI.getFrontendOpts().Inputs.emplace_back(Deps.ClangModuleMapFile,
                                           ModuleMapInputKind);

  auto CurrentModuleMapEntry =
      ScanInstance.getFileManager().getFile(Deps.ClangModuleMapFile);
  assert(CurrentModuleMapEntry && "module map file entry not found");

  auto DepModuleMapFiles = collectModuleMapFiles(Deps.ClangModuleDeps);
  for (StringRef ModuleMapFile : Deps.ModuleMapFileDeps) {
    // TODO: Track these as `FileEntryRef` to simplify the equality check below.
    auto ModuleMapEntry = ScanInstance.getFileManager().getFile(ModuleMapFile);
    assert(ModuleMapEntry && "module map file entry not found");

    // Don't report module maps describing eagerly-loaded dependency. This
    // information will be deserialized from the PCM.
    // TODO: Verify this works fine when modulemap for module A is eagerly
    // loaded from A.pcm, and module map passed on the command line contains
    // definition of a submodule: "explicit module A.Private { ... }".
    if (EagerLoadModules && DepModuleMapFiles.contains(*ModuleMapEntry))
      continue;

    // Don't report module map file of the current module unless it also
    // describes a dependency (for symmetry).
    if (*ModuleMapEntry == *CurrentModuleMapEntry &&
        !DepModuleMapFiles.contains(*ModuleMapEntry))
      continue;

    CI.getFrontendOpts().ModuleMapFiles.emplace_back(ModuleMapFile);
  }

  // Report the prebuilt modules this module uses.
  for (const auto &PrebuiltModule : Deps.PrebuiltModuleDeps)
    CI.getFrontendOpts().ModuleFiles.push_back(PrebuiltModule.PCMFile);

  // Add module file inputs from dependencies.
  addModuleFiles(CI, Deps.ClangModuleDeps);

  // Remove any macro definitions that are explicitly ignored.
  if (!CI.getHeaderSearchOpts().ModulesIgnoreMacros.empty()) {
    llvm::erase_if(
        CI.getPreprocessorOpts().Macros,
        [&CI](const std::pair<std::string, bool> &Def) {
          StringRef MacroDef = Def.first;
          return CI.getHeaderSearchOpts().ModulesIgnoreMacros.contains(
              llvm::CachedHashString(MacroDef.split('=').first));
        });
    // Remove the now unused option.
    CI.getHeaderSearchOpts().ModulesIgnoreMacros.clear();
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

llvm::DenseSet<const FileEntry *> ModuleDepCollector::collectModuleMapFiles(
    ArrayRef<ModuleID> ClangModuleDeps) const {
  llvm::DenseSet<const FileEntry *> ModuleMapFiles;
  for (const ModuleID &MID : ClangModuleDeps) {
    ModuleDeps *MD = ModuleDepsByID.lookup(MID);
    assert(MD && "Inconsistent dependency info");
    // TODO: Track ClangModuleMapFile as `FileEntryRef`.
    auto FE = ScanInstance.getFileManager().getFile(MD->ClangModuleMapFile);
    assert(FE && "Missing module map file that was previously found");
    ModuleMapFiles.insert(*FE);
  }
  return ModuleMapFiles;
}

void ModuleDepCollector::addModuleMapFiles(
    CompilerInvocation &CI, ArrayRef<ModuleID> ClangModuleDeps) const {
  if (EagerLoadModules)
    return; // Only pcm is needed for eager load.

  for (const ModuleID &MID : ClangModuleDeps) {
    ModuleDeps *MD = ModuleDepsByID.lookup(MID);
    assert(MD && "Inconsistent dependency info");
    CI.getFrontendOpts().ModuleMapFiles.push_back(MD->ClangModuleMapFile);
  }
}

void ModuleDepCollector::addModuleFiles(
    CompilerInvocation &CI, ArrayRef<ModuleID> ClangModuleDeps) const {
  for (const ModuleID &MID : ClangModuleDeps) {
    std::string PCMPath =
        Consumer.lookupModuleOutput(MID, ModuleOutputKind::ModuleFile);
    if (EagerLoadModules)
      CI.getFrontendOpts().ModuleFiles.push_back(std::move(PCMPath));
    else
      CI.getHeaderSearchOpts().PrebuiltModuleFiles.insert(
          {MID.ModuleName, std::move(PCMPath)});
  }
}

static bool needsModules(FrontendInputFile FIF) {
  switch (FIF.getKind().getLanguage()) {
  case Language::Unknown:
  case Language::Asm:
  case Language::LLVM_IR:
    return false;
  default:
    return true;
  }
}

void ModuleDepCollector::applyDiscoveredDependencies(CompilerInvocation &CI) {
  CI.clearImplicitModuleBuildOptions();

  if (llvm::any_of(CI.getFrontendOpts().Inputs, needsModules)) {
    Preprocessor &PP = ScanInstance.getPreprocessor();
    if (Module *CurrentModule = PP.getCurrentModuleImplementation())
      if (OptionalFileEntryRef CurrentModuleMap =
              PP.getHeaderSearchInfo()
                  .getModuleMap()
                  .getModuleMapFileForUniquing(CurrentModule))
        CI.getFrontendOpts().ModuleMapFiles.emplace_back(
            CurrentModuleMap->getName());

    SmallVector<ModuleID> DirectDeps;
    for (const auto &KV : ModularDeps)
      if (KV.second->ImportedByMainFile)
        DirectDeps.push_back(KV.second->ID);

    // TODO: Report module maps the same way it's done for modular dependencies.
    addModuleMapFiles(CI, DirectDeps);

    addModuleFiles(CI, DirectDeps);

    for (const auto &KV : DirectPrebuiltModularDeps)
      CI.getFrontendOpts().ModuleFiles.push_back(KV.second.PCMFile);
  }
}

static std::string getModuleContextHash(const ModuleDeps &MD,
                                        const CompilerInvocation &CI,
                                        bool EagerLoadModules) {
  llvm::HashBuilder<llvm::TruncatedBLAKE3<16>,
                    llvm::support::endianness::native>
      HashBuilder;
  SmallString<32> Scratch;

  // Hash the compiler version and serialization version to ensure the module
  // will be readable.
  HashBuilder.add(getClangFullRepositoryVersion());
  HashBuilder.add(serialization::VERSION_MAJOR, serialization::VERSION_MINOR);

  // Hash the BuildInvocation without any input files.
  SmallVector<const char *, 32> DummyArgs;
  CI.generateCC1CommandLine(DummyArgs, [&](const Twine &Arg) {
    Scratch.clear();
    StringRef Str = Arg.toStringRef(Scratch);
    HashBuilder.add(Str);
    return "<unused>";
  });

  // Hash the module dependencies. These paths may differ even if the invocation
  // is identical if they depend on the contents of the files in the TU -- for
  // example, case-insensitive paths to modulemap files. Usually such a case
  // would indicate a missed optimization to canonicalize, but it may be
  // difficult to canonicalize all cases when there is a VFS.
  for (const auto &ID : MD.ClangModuleDeps) {
    HashBuilder.add(ID.ModuleName);
    HashBuilder.add(ID.ContextHash);
  }

  HashBuilder.add(EagerLoadModules);

  llvm::BLAKE3Result<16> Hash = HashBuilder.final();
  std::array<uint64_t, 2> Words;
  static_assert(sizeof(Hash) == sizeof(Words), "Hash must match Words");
  std::memcpy(Words.data(), Hash.data(), sizeof(Hash));
  return toString(llvm::APInt(sizeof(Words) * 8, Words), 36, /*Signed=*/false);
}

void ModuleDepCollector::associateWithContextHash(const CompilerInvocation &CI,
                                                  ModuleDeps &Deps) {
  Deps.ID.ContextHash = getModuleContextHash(Deps, CI, EagerLoadModules);
  bool Inserted = ModuleDepsByID.insert({Deps.ID, &Deps}).second;
  (void)Inserted;
  assert(Inserted && "duplicate module mapping");
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
    MDC.ContextHash = MDC.ScanInstance.getInvocation().getModuleHash();
    MDC.Consumer.handleContextHash(MDC.ContextHash);
  }

  SourceManager &SM = MDC.ScanInstance.getSourceManager();

  // Dependency generation really does want to go all the way to the
  // file entry for a source location to find out what is depended on.
  // We do not want #line markers to affect dependency generation!
  if (std::optional<StringRef> Filename =
          SM.getNonBuiltinFilenameForID(SM.getFileID(SM.getExpansionLoc(Loc))))
    MDC.addFileDep(llvm::sys::path::remove_leading_dotslash(*Filename));
}

void ModuleDepCollectorPP::InclusionDirective(
    SourceLocation HashLoc, const Token &IncludeTok, StringRef FileName,
    bool IsAngled, CharSourceRange FilenameRange, OptionalFileEntryRef File,
    StringRef SearchPath, StringRef RelativePath, const Module *Imported,
    SrcMgr::CharacteristicKind FileType) {
  if (!File && !Imported) {
    // This is a non-modular include that HeaderSearch failed to find. Add it
    // here as `FileChanged` will never see it.
    MDC.addFileDep(FileName);
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
    MDC.DirectPrebuiltModularDeps.insert(
        {TopLevelModule, PrebuiltModuleDep{TopLevelModule}});
  else
    DirectModularDeps.insert(TopLevelModule);
}

void ModuleDepCollectorPP::EndOfMainFile() {
  FileID MainFileID = MDC.ScanInstance.getSourceManager().getMainFileID();
  MDC.MainFile = std::string(MDC.ScanInstance.getSourceManager()
                                 .getFileEntryForID(MainFileID)
                                 ->getName());

  if (!MDC.ScanInstance.getPreprocessorOpts().ImplicitPCHInclude.empty())
    MDC.addFileDep(MDC.ScanInstance.getPreprocessorOpts().ImplicitPCHInclude);

  for (const Module *M :
       MDC.ScanInstance.getPreprocessor().getAffectingClangModules())
    if (!MDC.isPrebuiltModule(M))
      DirectModularDeps.insert(M);

  for (const Module *M : DirectModularDeps)
    handleTopLevelModule(M);

  MDC.Consumer.handleDependencyOutputOpts(*MDC.Opts);

  for (auto &&I : MDC.ModularDeps)
    MDC.Consumer.handleModuleDependency(*I.second);

  for (auto &&I : MDC.FileDeps)
    MDC.Consumer.handleFileDependency(I);

  for (auto &&I : MDC.DirectPrebuiltModularDeps)
    MDC.Consumer.handlePrebuiltModuleDependency(I.second);
}

std::optional<ModuleID>
ModuleDepCollectorPP::handleTopLevelModule(const Module *M) {
  assert(M == M->getTopLevelModule() && "Expected top level module!");

  // A top-level module might not be actually imported as a module when
  // -fmodule-name is used to compile a translation unit that imports this
  // module. In that case it can be skipped. The appropriate header
  // dependencies will still be reported as expected.
  if (!M->getASTFile())
    return {};

  // If this module has been handled already, just return its ID.
  auto ModI = MDC.ModularDeps.insert({M, nullptr});
  if (!ModI.second)
    return ModI.first->second->ID;

  ModI.first->second = std::make_unique<ModuleDeps>();
  ModuleDeps &MD = *ModI.first->second;

  MD.ID.ModuleName = M->getFullModuleName();
  MD.ImportedByMainFile = DirectModularDeps.contains(M);
  MD.ImplicitModulePCMPath = std::string(M->getASTFile()->getName());
  MD.IsSystem = M->IsSystem;

  ModuleMap &ModMapInfo =
      MDC.ScanInstance.getPreprocessor().getHeaderSearchInfo().getModuleMap();

  OptionalFileEntryRef ModuleMap = ModMapInfo.getModuleMapFileForUniquing(M);

  if (ModuleMap) {
    SmallString<128> Path = ModuleMap->getNameAsRequested();
    ModMapInfo.canonicalizeModuleMapPath(Path);
    MD.ClangModuleMapFile = std::string(Path);
  }

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
          MDC.addFileDep(MD, ModuleMap->getName());
          return;
        }
        MDC.addFileDep(MD, IF.getFile()->getName());
      });

  llvm::DenseSet<const Module *> SeenDeps;
  addAllSubmodulePrebuiltDeps(M, MD, SeenDeps);
  addAllSubmoduleDeps(M, MD, SeenDeps);
  addAllAffectingClangModules(M, MD, SeenDeps);

  MDC.ScanInstance.getASTReader()->visitTopLevelModuleMaps(
      *MF, [&](FileEntryRef FE) {
        if (FE.getNameAsRequested().endswith("__inferred_module.map"))
          return;
        MD.ModuleMapFileDeps.emplace_back(FE.getNameAsRequested());
      });

  CompilerInvocation CI = MDC.makeInvocationForModuleBuildWithoutOutputs(
      MD, [&](CompilerInvocation &BuildInvocation) {
        if (MDC.OptimizeArgs)
          optimizeHeaderSearchOpts(BuildInvocation.getHeaderSearchOpts(),
                                   *MDC.ScanInstance.getASTReader(), *MF);
      });

  MDC.associateWithContextHash(CI, MD);

  // Finish the compiler invocation. Requires dependencies and the context hash.
  MDC.addOutputPaths(CI, MD);

  MD.BuildArguments = CI.getCC1CommandLine();

  return MD.ID;
}

static void forEachSubmoduleSorted(const Module *M,
                                   llvm::function_ref<void(const Module *)> F) {
  // Submodule order depends on order of header includes for inferred submodules
  // we don't care about the exact order, so sort so that it's consistent across
  // TUs to improve sharing.
  SmallVector<const Module *> Submodules(M->submodule_begin(),
                                         M->submodule_end());
  llvm::stable_sort(Submodules, [](const Module *A, const Module *B) {
    return A->Name < B->Name;
  });
  for (const Module *SubM : Submodules)
    F(SubM);
}

void ModuleDepCollectorPP::addAllSubmodulePrebuiltDeps(
    const Module *M, ModuleDeps &MD,
    llvm::DenseSet<const Module *> &SeenSubmodules) {
  addModulePrebuiltDeps(M, MD, SeenSubmodules);

  forEachSubmoduleSorted(M, [&](const Module *SubM) {
    addAllSubmodulePrebuiltDeps(SubM, MD, SeenSubmodules);
  });
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

  forEachSubmoduleSorted(M, [&](const Module *SubM) {
    addAllSubmoduleDeps(SubM, MD, AddedModules);
  });
}

void ModuleDepCollectorPP::addModuleDep(
    const Module *M, ModuleDeps &MD,
    llvm::DenseSet<const Module *> &AddedModules) {
  for (const Module *Import : M->Imports) {
    if (Import->getTopLevelModule() != M->getTopLevelModule() &&
        !MDC.isPrebuiltModule(Import)) {
      if (auto ImportID = handleTopLevelModule(Import->getTopLevelModule()))
        if (AddedModules.insert(Import->getTopLevelModule()).second)
          MD.ClangModuleDeps.push_back(*ImportID);
    }
  }
}

void ModuleDepCollectorPP::addAllAffectingClangModules(
    const Module *M, ModuleDeps &MD,
    llvm::DenseSet<const Module *> &AddedModules) {
  addAffectingClangModule(M, MD, AddedModules);

  for (const Module *SubM : M->submodules())
    addAllAffectingClangModules(SubM, MD, AddedModules);
}

void ModuleDepCollectorPP::addAffectingClangModule(
    const Module *M, ModuleDeps &MD,
    llvm::DenseSet<const Module *> &AddedModules) {
  for (const Module *Affecting : M->AffectingClangModules) {
    assert(Affecting == Affecting->getTopLevelModule() &&
           "Not quite import not top-level module");
    if (Affecting != M->getTopLevelModule() &&
        !MDC.isPrebuiltModule(Affecting)) {
      if (auto ImportID = handleTopLevelModule(Affecting))
        if (AddedModules.insert(Affecting).second)
          MD.ClangModuleDeps.push_back(*ImportID);
    }
  }
}

ModuleDepCollector::ModuleDepCollector(
    std::unique_ptr<DependencyOutputOptions> Opts,
    CompilerInstance &ScanInstance, DependencyConsumer &C,
    CompilerInvocation OriginalCI, bool OptimizeArgs, bool EagerLoadModules)
    : ScanInstance(ScanInstance), Consumer(C), Opts(std::move(Opts)),
      OriginalInvocation(std::move(OriginalCI)), OptimizeArgs(OptimizeArgs),
      EagerLoadModules(EagerLoadModules) {}

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

static StringRef makeAbsoluteAndPreferred(CompilerInstance &CI, StringRef Path,
                                          SmallVectorImpl<char> &Storage) {
  if (llvm::sys::path::is_absolute(Path) &&
      !llvm::sys::path::is_style_windows(llvm::sys::path::Style::native))
    return Path;
  Storage.assign(Path.begin(), Path.end());
  CI.getFileManager().makeAbsolutePath(Storage);
  llvm::sys::path::make_preferred(Storage);
  return StringRef(Storage.data(), Storage.size());
}

void ModuleDepCollector::addFileDep(StringRef Path) {
  llvm::SmallString<256> Storage;
  Path = makeAbsoluteAndPreferred(ScanInstance, Path, Storage);
  FileDeps.push_back(std::string(Path));
}

void ModuleDepCollector::addFileDep(ModuleDeps &MD, StringRef Path) {
  llvm::SmallString<256> Storage;
  Path = makeAbsoluteAndPreferred(ScanInstance, Path, Storage);
  MD.FileDeps.insert(Path);
}
