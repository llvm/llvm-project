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

static void updateCompilerInvocation(CompilerInvocation &Invocation,
                                     llvm::StringSaver &Saver,
                                     bool ProduceIncludeTree,
                                     std::string RootID,
                                     StringRef CASWorkingDirectory,
                                     llvm::PrefixMapper &Mapper) {
  // "Fix" the CAS options.
  auto &FileSystemOpts = Invocation.getFileSystemOpts();
  if (ProduceIncludeTree) {
    Invocation.getFrontendOpts().CASIncludeTreeID = RootID;
    Invocation.getFrontendOpts().Inputs.clear();
    // Preserve sysroot path to accommodate lookup for 'SDKSettings.json' during
    // availability checking.
    std::string OriginalSysroot = Invocation.getHeaderSearchOpts().Sysroot;
    Invocation.getHeaderSearchOpts() = HeaderSearchOptions();
    Invocation.getHeaderSearchOpts().Sysroot = OriginalSysroot;
    auto &PPOpts = Invocation.getPreprocessorOpts();
    // We don't need this because we save the contents of the PCH file in the
    // include tree root.
    PPOpts.ImplicitPCHInclude.clear();
    if (Invocation.getFrontendOpts().ProgramAction != frontend::GeneratePCH) {
      // We don't need these because we save the contents of the predefines
      // buffer in the include tree. But if we generate a PCH file we still need
      // to keep them as preprocessor options so that they are preserved in a
      // PCH file and compared with the preprocessor options of the dep-scan
      // invocation that uses the PCH.
      PPOpts.Macros.clear();
      PPOpts.MacroIncludes.clear();
      PPOpts.Includes.clear();
    }
  } else {
    FileSystemOpts.CASFileSystemRootID = RootID;
    FileSystemOpts.CASFileSystemWorkingDirectory = CASWorkingDirectory.str();
  }
  auto &FrontendOpts = Invocation.getFrontendOpts();
  FrontendOpts.CacheCompileJob = true; // FIXME: Don't set.

  // Turn off dependency outputs. Should have already been emitted.
  Invocation.getDependencyOutputOpts().OutputFile.clear();

  // Apply path remappings.
  DepscanPrefixMapping::remapInvocationPaths(Invocation, Mapper);
}

void DepscanPrefixMapping::remapInvocationPaths(CompilerInvocation &Invocation,
                                                llvm::PrefixMapper &Mapper) {
  // If there are no mappings, we're done. Otherwise, continue and remap
  // everything.
  if (Mapper.getMappings().empty())
    return;

  // Returns "false" on success, "true" if the path doesn't exist.
  auto remapInPlace = [&](std::string &S) -> bool {
    return errorToBool(Mapper.mapInPlace(S));
  };

  auto remapInPlaceOrFilterOutWith = [&](auto &Vector, auto Remapper) {
    Vector.erase(llvm::remove_if(Vector, Remapper), Vector.end());
  };

  auto remapInPlaceOrFilterOut = [&](std::vector<std::string> &Vector) {
    remapInPlaceOrFilterOutWith(Vector, remapInPlace);
  };

  // If we can't remap the working directory, skip everything else.
  auto &FileSystemOpts = Invocation.getFileSystemOpts();
  if (remapInPlace(FileSystemOpts.CASFileSystemWorkingDirectory))
    return;

  // Remap header search.
  auto &HeaderSearchOpts = Invocation.getHeaderSearchOpts();
  Mapper.mapInPlaceOrClear(HeaderSearchOpts.Sysroot);
  remapInPlaceOrFilterOutWith(HeaderSearchOpts.UserEntries,
                              [&](HeaderSearchOptions::Entry &Entry) {
                                if (Entry.IgnoreSysRoot)
                                  return remapInPlace(Entry.Path);
                                return false;
                              });
  remapInPlaceOrFilterOutWith(
      HeaderSearchOpts.SystemHeaderPrefixes,
      [&](HeaderSearchOptions::SystemHeaderPrefix &Prefix) {
        return remapInPlace(Prefix.Prefix);
      });
  Mapper.mapInPlaceOrClear(HeaderSearchOpts.ResourceDir);
  Mapper.mapInPlaceOrClear(HeaderSearchOpts.ModuleCachePath);
  Mapper.mapInPlaceOrClear(HeaderSearchOpts.ModuleUserBuildPath);
  for (auto I = HeaderSearchOpts.PrebuiltModuleFiles.begin(),
            E = HeaderSearchOpts.PrebuiltModuleFiles.end();
       I != E;) {
    auto Current = I++;
    if (remapInPlace(Current->second))
      HeaderSearchOpts.PrebuiltModuleFiles.erase(Current);
  }
  remapInPlaceOrFilterOut(HeaderSearchOpts.PrebuiltModulePaths);
  remapInPlaceOrFilterOut(HeaderSearchOpts.VFSOverlayFiles);

  // Preprocessor options.
  auto &PPOpts = Invocation.getPreprocessorOpts();
  remapInPlaceOrFilterOut(PPOpts.MacroIncludes);
  remapInPlaceOrFilterOut(PPOpts.Includes);
  Mapper.mapInPlaceOrClear(PPOpts.ImplicitPCHInclude);

  // Frontend options.
  auto &FrontendOpts = Invocation.getFrontendOpts();
  remapInPlaceOrFilterOutWith(
      FrontendOpts.Inputs, [&](FrontendInputFile &Input) {
        if (Input.isBuffer())
          return false; // FIXME: Can this happen when parsing command-line?

        SmallString<256> PathBuf;
        Optional<StringRef> RemappedFile =
            Mapper.mapOrNoneIfError(Input.getFile(), PathBuf);
        if (!RemappedFile)
          return true;
        if (RemappedFile != Input.getFile())
          Input = FrontendInputFile(*RemappedFile, Input.getKind(),
                                    Input.isSystem());
        return false;
      });

  // Skip the output file. That's not the input CAS filesystem.
  //   Mapper.mapInPlaceOrClear(OutputFile); <-- this doesn't make sense.

  Mapper.mapInPlaceOrClear(FrontendOpts.CodeCompletionAt.FileName);

  // Don't remap plugins (for now), since we don't know how to remap their
  // arguments. Maybe they should be loaded outside of the CAS filesystem?
  // Maybe we should error?
  //
  //  remapInPlaceOrFilterOut(FrontendOpts.Plugins);

  remapInPlaceOrFilterOut(FrontendOpts.ModuleMapFiles);
  remapInPlaceOrFilterOut(FrontendOpts.ModuleFiles);
  remapInPlaceOrFilterOut(FrontendOpts.ModulesEmbedFiles);
  remapInPlaceOrFilterOut(FrontendOpts.ASTMergeFiles);
  Mapper.mapInPlaceOrClear(FrontendOpts.OverrideRecordLayoutsFile);
  Mapper.mapInPlaceOrClear(FrontendOpts.StatsFile);

  // Filesystem options.
  Mapper.mapInPlaceOrClear(FileSystemOpts.WorkingDir);

  // Code generation options.
  auto &CodeGenOpts = Invocation.getCodeGenOpts();
  Mapper.mapInPlaceOrClear(CodeGenOpts.DebugCompilationDir);
  Mapper.mapInPlaceOrClear(CodeGenOpts.CoverageCompilationDir);

  // Handle coverage mappings.
  Mapper.mapInPlaceOrClear(CodeGenOpts.ProfileInstrumentUsePath);
  Mapper.mapInPlaceOrClear(CodeGenOpts.SampleProfileFile);
  Mapper.mapInPlaceOrClear(CodeGenOpts.ProfileRemappingFile);
}

Error DepscanPrefixMapping::configurePrefixMapper(
    const CompilerInvocation &Invocation, llvm::StringSaver &Saver,
    llvm::PrefixMapper &Mapper) const {
  auto isPathApplicableAsPrefix = [](StringRef Path) -> bool {
    if (Path.empty())
      return false;
    if (llvm::sys::path::is_relative(Path))
      return false;
    if (Path == llvm::sys::path::root_path(Path))
      return false;
    return true;
  };

  const HeaderSearchOptions &HSOpts = Invocation.getHeaderSearchOpts();

  if (NewSDKPath) {
    StringRef SDK = HSOpts.Sysroot;
    if (isPathApplicableAsPrefix(SDK))
      // Need a new copy of the string since the invocation will be modified.
      if (auto E = Mapper.add({Saver.save(SDK), *NewSDKPath}))
        return E;
  }
  if (NewToolchainPath) {
    // Look up for the toolchain, assuming resources are at
    // <toolchain>/usr/lib/clang/<VERSION>. Return a shallower guess if the
    // directories do not match.
    //
    // FIXME: Should this append ".." instead of calling parent_path?
    StringRef ResourceDir = HSOpts.ResourceDir;
    StringRef Guess = llvm::sys::path::parent_path(ResourceDir);
    for (StringRef Dir : {"clang", "lib", "usr"}) {
      if (llvm::sys::path::filename(Guess) != Dir)
        break;
      Guess = llvm::sys::path::parent_path(Guess);
    }
    if (isPathApplicableAsPrefix(Guess))
      // Need a new copy of the string since the invocation will be modified.
      if (auto E = Mapper.add({Saver.save(Guess), *NewToolchainPath}))
        return E;
  }
  if (!PrefixMap.empty()) {
    llvm::SmallVector<llvm::MappedPrefix> Split;
    llvm::MappedPrefix::transformJoinedIfValid(PrefixMap, Split);
    for (auto &MappedPrefix : Split) {
      if (isPathApplicableAsPrefix(MappedPrefix.Old)) {
        if (auto E = Mapper.add(MappedPrefix))
          return E;
      } else {
        return createStringError(llvm::errc::invalid_argument,
                                 "invalid prefix map: '" + MappedPrefix.Old +
                                     "=" + MappedPrefix.New + "'");
      }
    }
  }

  Mapper.sort();
  return Error::success();
}

Expected<llvm::cas::CASID> clang::scanAndUpdateCC1InlineWithTool(
    DependencyScanningTool &Tool, DiagnosticConsumer &DiagsConsumer,
    raw_ostream *VerboseOS, CompilerInvocation &Invocation,
    StringRef WorkingDirectory, const DepscanPrefixMapping &PrefixMapping,
    llvm::cas::ObjectStore &DB) {
  llvm::cas::CachingOnDiskFileSystem &FS = Tool.getCachingFileSystem();

  // Override the CASOptions. They may match (the caller having sniffed them
  // out of InputArgs) but if they have been overridden we want the new ones.
  Invocation.getCASOpts() = Tool.getCASOpts();

  llvm::BumpPtrAllocator Alloc;
  llvm::StringSaver Saver(Alloc);
  llvm::TreePathPrefixMapper Mapper(&FS);
  if (Error E = PrefixMapping.configurePrefixMapper(Invocation, Saver, Mapper))
    return std::move(E);

  auto ScanInvocation = std::make_shared<CompilerInvocation>(Invocation);
  // An error during dep-scanning is treated as if the main compilation has
  // failed, but warnings are ignored and deferred for the main compilation.
  ScanInvocation->getDiagnosticOpts().IgnoreWarnings = true;

  Optional<llvm::cas::CASID> Root;
  bool ProduceIncludeTree =
      Tool.getScanningFormat() ==
      tooling::dependencies::ScanningOutputFormat::IncludeTree;
  if (ProduceIncludeTree) {
    if (Error E = Tool.getIncludeTreeFromCompilerInvocation(
                          DB, std::move(ScanInvocation), WorkingDirectory,
                          DiagsConsumer, VerboseOS,
                          /*DiagGenerationAsCompilation*/ true)
                      .moveInto(Root))
      return std::move(E);
  } else {
    if (Error E = Tool.getDependencyTreeFromCompilerInvocation(
                          std::move(ScanInvocation), WorkingDirectory,
                          DiagsConsumer, VerboseOS,
                          /*DiagGenerationAsCompilation*/ true,
                          [&](const llvm::vfs::CachedDirectoryEntry &Entry) {
                            return Mapper.mapDirEntry(Entry, Saver);
                          })
                      .moveInto(Root))
      return std::move(E);
  }
  updateCompilerInvocation(Invocation, Saver, ProduceIncludeTree,
                           Root->toString(), WorkingDirectory, Mapper);
  return *Root;
}
