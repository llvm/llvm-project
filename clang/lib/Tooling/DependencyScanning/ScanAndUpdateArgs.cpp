//===- ScanAndUpdateArgs.cpp - Util for CC1 Dependency Scanning -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Tooling/DependencyScanning/ScanAndUpdateArgs.h"
#include "clang/CAS/IncludeTree.h"
#include "clang/Frontend/CompilerInvocation.h"
#include "clang/Lex/HeaderSearchOptions.h"
#include "clang/Tooling/DependencyScanning/DependencyScanningTool.h"
#include "llvm/CAS/CachingOnDiskFileSystem.h"
#include "llvm/CAS/ObjectStore.h"
#include "llvm/Support/PrefixMapper.h"

using namespace clang;
using namespace clang::tooling::dependencies;
using llvm::Error;

void tooling::dependencies::configureInvocationForCaching(
    CompilerInvocation &CI, CASOptions CASOpts, std::string RootID,
    std::string WorkingDir, bool ProduceIncludeTree) {
  CI.getCASOpts() = std::move(CASOpts);
  auto &FrontendOpts = CI.getFrontendOpts();
  FrontendOpts.CacheCompileJob = true;
  FrontendOpts.IncludeTimestamps = false;
  // Clear this otherwise it defeats the purpose of making the compilation key
  // independent of certain arguments.
  CI.getCodeGenOpts().DwarfDebugFlags.clear();
  if (FrontendOpts.ProgramAction == frontend::GeneratePCH) {
    // Clear paths that are emitted into binaries; they do not affect PCH.
    // For modules this is handled in ModuleDepCollector.
    CI.getCodeGenOpts().CoverageDataFile.clear();
    CI.getCodeGenOpts().CoverageNotesFile.clear();
  }

  HeaderSearchOptions &HSOpts = CI.getHeaderSearchOpts();
  // Avoid writing potentially volatile diagnostic options into pcms.
  HSOpts.ModulesSkipDiagnosticOptions = true;

  // "Fix" the CAS options.
  auto &FileSystemOpts = CI.getFileSystemOpts();
  if (ProduceIncludeTree) {
    FrontendOpts.CASIncludeTreeID = std::move(RootID);
    FrontendOpts.Inputs.clear();
    FrontendOpts.ModuleMapFiles.clear();
    HeaderSearchOptions OriginalHSOpts;
    std::swap(HSOpts, OriginalHSOpts);
    HSOpts.ModulesSkipDiagnosticOptions =
        OriginalHSOpts.ModulesSkipDiagnosticOptions;
    // Preserve sysroot path to accommodate lookup for 'SDKSettings.json' during
    // availability checking.
    HSOpts.Sysroot = std::move(OriginalHSOpts.Sysroot);
    // Preserve resource-dir, which is added back by cc1_main if missing, and
    // affects the cache key.
    HSOpts.ResourceDir = std::move(OriginalHSOpts.ResourceDir);
    // Preserve fmodule-file options.
    HSOpts.PrebuiltModuleFiles = std::move(OriginalHSOpts.PrebuiltModuleFiles);
    auto &PPOpts = CI.getPreprocessorOpts();
    // We don't need this because we save the contents of the PCH file in the
    // include tree root.
    PPOpts.ImplicitPCHInclude.clear();
    if (FrontendOpts.ProgramAction != frontend::GeneratePCH) {
      // We don't need these because we save the contents of the predefines
      // buffer in the include tree. But if we generate a PCH file we still need
      // to keep them as preprocessor options so that they are preserved in a
      // PCH file and compared with the preprocessor options of the dep-scan
      // invocation that uses the PCH.
      PPOpts.Macros.clear();
      PPOpts.MacroIncludes.clear();
      PPOpts.Includes.clear();
    }
    // Disable `-gmodules` to avoid debug info referencing a non-existent PCH
    // filename.
    // NOTE: We'd have to preserve \p HeaderSearchOptions::ModuleFormat (code
    // above resets \p HeaderSearchOptions) when properly supporting
    // `-gmodules`.
    CI.getCodeGenOpts().DebugTypeExtRefs = false;
    // Clear APINotes options.
    CI.getAPINotesOpts().ModuleSearchPaths = {};
  } else {
    FileSystemOpts.CASFileSystemRootID = std::move(RootID);
    FileSystemOpts.CASFileSystemWorkingDirectory = std::move(WorkingDir);
  }
}

void DepscanPrefixMapping::remapInvocationPaths(CompilerInvocation &Invocation,
                                                llvm::PrefixMapper &Mapper) {
  auto &FrontendOpts = Invocation.getFrontendOpts();
  FrontendOpts.PathPrefixMappings.clear();

  // If there are no mappings, we're done. Otherwise, continue and remap
  // everything.
  if (Mapper.empty())
    return;

  // Pass the remappings so that we can map cached diagnostics to the local
  // paths during diagnostic rendering.
  for (const llvm::MappedPrefix &Map : Mapper.getMappings()) {
    FrontendOpts.PathPrefixMappings.push_back(Map.Old + "=" + Map.New);
  }

  auto mapInPlaceAll = [&](std::vector<std::string> &Vector) {
    for (auto &Path : Vector)
      Mapper.mapInPlace(Path);
  };

  auto &FileSystemOpts = Invocation.getFileSystemOpts();
  Mapper.mapInPlace(FileSystemOpts.CASFileSystemWorkingDirectory);

  // Remap header search.
  auto &HeaderSearchOpts = Invocation.getHeaderSearchOpts();
  Mapper.mapInPlace(HeaderSearchOpts.Sysroot);
  for (auto &Entry : HeaderSearchOpts.UserEntries)
    if (Entry.IgnoreSysRoot)
      Mapper.mapInPlace(Entry.Path);

  for (auto &Prefix : HeaderSearchOpts.SystemHeaderPrefixes)
    Mapper.mapInPlace(Prefix.Prefix);
  Mapper.mapInPlace(HeaderSearchOpts.ResourceDir);
  Mapper.mapInPlace(HeaderSearchOpts.ModuleCachePath);
  Mapper.mapInPlace(HeaderSearchOpts.ModuleUserBuildPath);
  for (auto I = HeaderSearchOpts.PrebuiltModuleFiles.begin(),
            E = HeaderSearchOpts.PrebuiltModuleFiles.end();
       I != E;) {
    auto Current = I++;
    Mapper.mapInPlace(Current->second);
  }
  mapInPlaceAll(HeaderSearchOpts.PrebuiltModulePaths);
  mapInPlaceAll(HeaderSearchOpts.VFSOverlayFiles);

  // Preprocessor options.
  auto &PPOpts = Invocation.getPreprocessorOpts();
  mapInPlaceAll(PPOpts.MacroIncludes);
  mapInPlaceAll(PPOpts.Includes);
  Mapper.mapInPlace(PPOpts.ImplicitPCHInclude);

  // Frontend options.
  for (FrontendInputFile &Input : FrontendOpts.Inputs) {
    if (Input.isBuffer())
      continue; // FIXME: Can this happen when parsing command-line?

    SmallString<256> RemappedFile;
    Mapper.map(Input.getFile(), RemappedFile);
    if (RemappedFile != Input.getFile())
      Input =
          FrontendInputFile(RemappedFile, Input.getKind(), Input.isSystem());
  }

  // Skip the output file. That's not the input CAS filesystem.
  //   Mapper.mapInPlace(OutputFile); <-- this doesn't make sense.

  Mapper.mapInPlace(FrontendOpts.CodeCompletionAt.FileName);

  // Don't remap plugins (for now), since we don't know how to remap their
  // arguments. Maybe they should be loaded outside of the CAS filesystem?
  // Maybe we should error?
  //
  //  Mapper.mapInPlaceOrFilterOut(FrontendOpts.Plugins);

  mapInPlaceAll(FrontendOpts.ModuleMapFiles);
  mapInPlaceAll(FrontendOpts.ModuleFiles);
  mapInPlaceAll(FrontendOpts.ModulesEmbedFiles);
  mapInPlaceAll(FrontendOpts.ASTMergeFiles);
  Mapper.mapInPlace(FrontendOpts.OverrideRecordLayoutsFile);
  Mapper.mapInPlace(FrontendOpts.StatsFile);
  for (auto &[Path, _] : FrontendOpts.ModuleCacheKeys)
    Mapper.mapInPlace(Path);

  // Filesystem options.
  Mapper.mapInPlace(FileSystemOpts.WorkingDir);

  // Code generation options.
  auto &CodeGenOpts = Invocation.getCodeGenOpts();
  Mapper.mapInPlace(CodeGenOpts.DebugCompilationDir);
  Mapper.mapInPlace(CodeGenOpts.CoverageCompilationDir);

  // Sanitizer options.
  mapInPlaceAll(Invocation.getLangOpts().NoSanitizeFiles);

  // Handle coverage mappings.
  Mapper.mapInPlace(CodeGenOpts.ProfileInstrumentUsePath);
  Mapper.mapInPlace(CodeGenOpts.SampleProfileFile);
  Mapper.mapInPlace(CodeGenOpts.ProfileRemappingFile);

  // Dependency output options.
  // Note: these are not in the cache key, but they are in the module context
  // hash, which indirectly impacts the cache key when importing a module.
  // In the future we may change how -fmodule-file-cache-key works when
  // remapping to avoid needing this.
  for (auto &ExtraDep : Invocation.getDependencyOutputOpts().ExtraDeps)
    Mapper.mapInPlace(ExtraDep.first);
}

void DepscanPrefixMapping::configurePrefixMapper(const CompilerInvocation &CI,
                                                 llvm::PrefixMapper &Mapper) {
  return configurePrefixMapper(CI.getFrontendOpts().PathPrefixMappings, Mapper);
}

void DepscanPrefixMapping::configurePrefixMapper(
    ArrayRef<std::string> PathPrefixMappings, llvm::PrefixMapper &Mapper) {
  if (PathPrefixMappings.empty())
    return;

  llvm::SmallVector<llvm::MappedPrefix> Split;
  llvm::MappedPrefix::transformJoinedIfValid(PathPrefixMappings, Split);
  for (auto &MappedPrefix : Split)
    Mapper.add(MappedPrefix);

  Mapper.sort();
}

Expected<llvm::cas::CASID> clang::scanAndUpdateCC1InlineWithTool(
    DependencyScanningTool &Tool, DiagnosticConsumer &DiagsConsumer,
    raw_ostream *VerboseOS, CompilerInvocation &Invocation,
    StringRef WorkingDirectory, llvm::cas::ObjectStore &DB) {
  // Override the CASOptions. They may match (the caller having sniffed them
  // out of InputArgs) but if they have been overridden we want the new ones.
  Invocation.getCASOpts() = Tool.getCASOpts();

  bool ProduceIncludeTree =
      Tool.getScanningFormat() ==
      tooling::dependencies::ScanningOutputFormat::IncludeTree;

  std::unique_ptr<llvm::PrefixMapper> MapperPtr;
  if (ProduceIncludeTree) {
    MapperPtr = std::make_unique<llvm::PrefixMapper>();
  } else {
    MapperPtr = std::make_unique<llvm::TreePathPrefixMapper>(
        Tool.getCachingFileSystem());
  }
  llvm::PrefixMapper &Mapper = *MapperPtr;
  DepscanPrefixMapping::configurePrefixMapper(Invocation, Mapper);

  auto ScanInvocation = std::make_shared<CompilerInvocation>(Invocation);
  // An error during dep-scanning is treated as if the main compilation has
  // failed, but warnings are ignored and deferred for the main compilation.
  ScanInvocation->getDiagnosticOpts().IgnoreWarnings = true;

  std::optional<llvm::cas::CASID> Root;
  if (ProduceIncludeTree) {
    if (Error E =
            Tool.getIncludeTreeFromCompilerInvocation(
                    DB, std::move(ScanInvocation), WorkingDirectory,
                    /*LookupModuleOutput=*/nullptr, DiagsConsumer, VerboseOS,
                    /*DiagGenerationAsCompilation*/ true)
                .moveInto(Root))
      return std::move(E);
  } else {
    if (Error E = Tool.getDependencyTreeFromCompilerInvocation(
                          std::move(ScanInvocation), WorkingDirectory,
                          DiagsConsumer, VerboseOS,
                          /*DiagGenerationAsCompilation*/ true)
                      .moveInto(Root))
      return std::move(E);
  }

  // Turn off dependency outputs. Should have already been emitted.
  Invocation.getDependencyOutputOpts().OutputFile.clear();

  configureInvocationForCaching(Invocation, Tool.getCASOpts(), Root->toString(),
                                WorkingDirectory.str(), ProduceIncludeTree);
  DepscanPrefixMapping::remapInvocationPaths(Invocation, Mapper);
  return *Root;
}
