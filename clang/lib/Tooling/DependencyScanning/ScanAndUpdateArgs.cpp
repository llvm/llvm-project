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
#include "llvm/CAS/ObjectStore.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/PrefixMapper.h"

using namespace clang;
using namespace clang::tooling::dependencies;
using llvm::Error;

static void updateRelativePath(std::string &Path,
                               const std::string &WorkingDir) {
  if (Path.empty() || llvm::sys::path::is_absolute(Path) || WorkingDir.empty())
    return;

  SmallString<128> PathStorage(WorkingDir);
  llvm::sys::path::append(PathStorage, Path);
  Path = PathStorage.str();
}

void tooling::dependencies::configureInvocationForCaching(
    CompilerInvocation &CI, CASOptions CASOpts, std::string InputID,
    CachingInputKind InputKind, std::string WorkingDir) {
  CI.getCASOpts() = std::move(CASOpts);
  auto &FrontendOpts = CI.getFrontendOpts();
  FrontendOpts.CacheCompileJob = true;
  FrontendOpts.IncludeTimestamps = false;

  // Clear this otherwise it defeats the purpose of making the compilation key
  // independent of certain arguments.
  auto &CodeGenOpts = CI.getCodeGenOpts();
  if (CI.getFrontendOpts().ProgramAction != frontend::ActionKind::EmitObj) {
    CodeGenOpts.UseCASBackend = false;
    CodeGenOpts.EmitCASIDFile = false;
    auto &LLVMArgs = FrontendOpts.LLVMArgs;
    llvm::erase(LLVMArgs, "-cas-friendly-debug-info");
  }
  CodeGenOpts.DwarfDebugFlags.clear();
  resetBenignCodeGenOptions(FrontendOpts.ProgramAction, CI.getLangOpts(),
                            CodeGenOpts);

  HeaderSearchOptions &HSOpts = CI.getHeaderSearchOpts();
  // Avoid writing potentially volatile diagnostic options into pcms.
  HSOpts.ModulesSkipDiagnosticOptions = true;

  // "Fix" the CAS options.
  auto &FileSystemOpts = CI.getFileSystemOpts();
  switch (InputKind) {
  case CachingInputKind::IncludeTree: {
    using llvm::sys::path::is_relative;
    // The only way to get relative source paths is a relative main file or
    // include directory, since includes are resolved relative to , and
    // includer-relative search is relative to an already-included file.
    bool HasRelativeIncludes = [&]{
      if (!FileSystemOpts.WorkingDir.empty())
        return false; // -working-directory makes all paths absolute.
      // FIXME: this might have false positives if a relative search path is not
      // used. Ideally we would calculate this when the include tree is built.
      if (!HSOpts.Sysroot.empty() && is_relative(HSOpts.Sysroot))
        return true;
      if (is_relative(HSOpts.ResourceDir))
        return true;
      for (auto &Input : FrontendOpts.Inputs)
        if (Input.isFile() && is_relative(Input.getFile()))
          return true;
      for (auto &Entry : HSOpts.UserEntries)
        if (Entry.IgnoreSysRoot || HSOpts.Sysroot.empty())
          if (is_relative(Entry.Path))
            return true;
      for (auto &Prefix : HSOpts.SystemHeaderPrefixes)
        if (is_relative(Prefix.Prefix))
          return true;
      return false;
    }();

    FrontendOpts.CASIncludeTreeID = std::move(InputID);
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
    // Preserve -gmodules (see below for caveats).
    HSOpts.ModuleFormat = OriginalHSOpts.ModuleFormat;
    HSOpts.UseBuiltinIncludes = false;
    HSOpts.UseStandardSystemIncludes = false;
    HSOpts.UseStandardCXXIncludes = false;

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
    // Clear APINotes options.
    CI.getAPINotesOpts().ModuleSearchPaths = {};

    // Reset debug compilation directory if source paths are absolute.
    if (!HasRelativeIncludes) {
      CodeGenOpts.DebugCompilationDir.clear();
    }

    // Update output paths, and clear working directory.
    auto CWD = FileSystemOpts.WorkingDir;
    updateRelativePath(FrontendOpts.OutputFile, CWD);
    updateRelativePath(CI.getDiagnosticOpts().DiagnosticSerializationFile, CWD);
    updateRelativePath(CI.getDiagnosticOpts().DiagnosticLogFile, CWD);
    updateRelativePath(CI.getDependencyOutputOpts().OutputFile, CWD);
    FileSystemOpts.WorkingDir.clear();
    break;
  }
  case CachingInputKind::CachedCompilation: {
    FrontendOpts.Inputs.clear();
    FrontendOpts.CASInputFileCacheKey = std::move(InputID);
    FrontendOpts.ModuleMapFiles.clear();
    // TODO: Strip more of the things we already strip for include-tree.
    break;
  case CachingInputKind::Object: {
    assert(false && "Object should not be available during scanning");
  }
  }
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
    FrontendOpts.PathPrefixMappings.emplace_back(Map.Old, Map.New);
  }

  auto mapInPlaceAll = [&](std::vector<std::string> &Vector) {
    for (auto &Path : Vector)
      Mapper.mapInPlace(Path);
  };

  auto &FileSystemOpts = Invocation.getFileSystemOpts();

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
    ArrayRef<std::pair<std::string, std::string>> PathPrefixMappings, llvm::PrefixMapper &Mapper) {
  if (PathPrefixMappings.empty())
    return;

  llvm::SmallVector<llvm::MappedPrefix> Split;
  llvm::MappedPrefix::transformPairs(PathPrefixMappings, Split);
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

  llvm::PrefixMapper Mapper;
  DepscanPrefixMapping::configurePrefixMapper(Invocation, Mapper);

  auto ScanInvocation = std::make_shared<CompilerInvocation>(Invocation);
  // An error during dep-scanning is treated as if the main compilation has
  // failed, but warnings are ignored and deferred for the main compilation.
  ScanInvocation->getDiagnosticOpts().IgnoreWarnings = true;

  LookupModuleOutputCallback Lookup;

  std::optional<llvm::cas::CASID> Root;
  if (Error E =
          Tool.getIncludeTreeFromCompilerInvocation(
                  DB, std::move(ScanInvocation), WorkingDirectory,
                  /*LookupModuleOutput=*/nullptr, DiagsConsumer, VerboseOS,
                  /*DiagGenerationAsCompilation*/ true)
              .moveInto(Root))
    return std::move(E);

  // Turn off dependency outputs. Should have already been emitted.
  Invocation.getDependencyOutputOpts().OutputFile.clear();

  configureInvocationForCaching(Invocation, Tool.getCASOpts(), Root->toString(),
                                CachingInputKind::IncludeTree,
                                WorkingDirectory.str());
  DepscanPrefixMapping::remapInvocationPaths(Invocation, Mapper);
  return *Root;
}
