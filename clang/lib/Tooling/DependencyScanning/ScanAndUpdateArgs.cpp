//===- ScanAndUpdateArgs.cpp - Util for CC1 Dependency Scanning -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Frontend/CompilerInvocation.h"
#include "clang/Lex/HeaderSearchOptions.h"
#include "clang/Tooling/DependencyScanning/DependencyScanningTool.h"
#include "clang/Tooling/DependencyScanning/ScanAndUpdateArgs.h"
#include "llvm/CAS/CASDB.h"
#include "llvm/CAS/CachingOnDiskFileSystem.h"
#include "llvm/Support/PrefixMapper.h"

using namespace clang;
using llvm::Error;

static Error computeSDKMapping(llvm::StringSaver &Saver,
                               const CompilerInvocation &Invocation,
                               StringRef New,
                               llvm::TreePathPrefixMapper &Mapper) {
  StringRef SDK = Invocation.getHeaderSearchOpts().Sysroot;
  if (SDK.empty())
    return Error::success();

  // Need a new copy of the string since the invocation will be modified.
  return Mapper.add(llvm::MappedPrefix{Saver.save(SDK), New});
}

static Error computeToolchainMapping(llvm::StringSaver &Saver,
                                     StringRef ClangPath, StringRef New,
                                     llvm::TreePathPrefixMapper &Mapper) {
  // Look up from clang for the toolchain, assuming clang is at
  // <toolchain>/usr/bin/clang. Return a shallower guess if the directories
  // don't match.
  //
  // FIXME: Should this append ".." instead of calling parent_path?
  StringRef Guess = llvm::sys::path::parent_path(ClangPath);
  for (StringRef Dir : {"bin", "usr"}) {
    if (llvm::sys::path::filename(Guess) != Dir)
      break;
    Guess = llvm::sys::path::parent_path(Guess);
  }
  return Mapper.add(llvm::MappedPrefix{Guess, New});
}

static Error
computeFullMapping(llvm::StringSaver &Saver, StringRef ClangPath,
                   const CompilerInvocation &Invocation,
                   const cc1depscand::DepscanPrefixMapping &DepscanMapping,
                   llvm::TreePathPrefixMapper &Mapper) {
  if (DepscanMapping.NewSDKPath)
    if (Error E = computeSDKMapping(Saver, Invocation,
                                    *DepscanMapping.NewSDKPath, Mapper))
      return E;

  if (DepscanMapping.NewToolchainPath)
    if (Error E = computeToolchainMapping(
            Saver, ClangPath, *DepscanMapping.NewToolchainPath, Mapper))
      return E;

  if (!DepscanMapping.PrefixMap.empty()) {
    llvm::SmallVector<llvm::MappedPrefix> Split;
    llvm::MappedPrefix::transformJoinedIfValid(DepscanMapping.PrefixMap, Split);
    if (Error E = Mapper.addRange(Split))
      return E;
  }

  Mapper.sort();
  return Error::success();
}

static void updateCompilerInvocation(CompilerInvocation &Invocation,
                                     llvm::StringSaver &Saver,
                                     llvm::cas::CachingOnDiskFileSystem &FS,
                                     std::string RootID,
                                     StringRef CASWorkingDirectory,
                                     llvm::TreePathPrefixMapper &Mapper) {
  // "Fix" the CAS options.
  auto &FileSystemOpts = Invocation.getFileSystemOpts();
  FileSystemOpts.CASFileSystemRootID = RootID;
  FileSystemOpts.CASFileSystemWorkingDirectory = CASWorkingDirectory.str();
  auto &FrontendOpts = Invocation.getFrontendOpts();
  FrontendOpts.CacheCompileJob = true; // FIXME: Don't set.

  // Turn off dependency outputs. Should have already been emitted.
  Invocation.getDependencyOutputOpts().OutputFile.clear();

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

  // Frontend options.
  remapInPlaceOrFilterOutWith(
      FrontendOpts.Inputs, [&](FrontendInputFile &Input) {
        if (Input.isBuffer())
          return false; // FIXME: Can this happen when parsing command-line?

        Optional<StringRef> RemappedFile = Mapper.mapOrNone(Input.getFile());
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
}

Expected<llvm::cas::CASID> clang::scanAndUpdateCC1InlineWithTool(
    tooling::dependencies::DependencyScanningTool &Tool,
    DiagnosticConsumer &DiagsConsumer, const char *Exec,
    CompilerInvocation &Invocation, StringRef WorkingDirectory,
    const cc1depscand::DepscanPrefixMapping &PrefixMapping) {
  llvm::cas::CachingOnDiskFileSystem &FS = Tool.getCachingFileSystem();

  // Override the CASOptions. They may match (the caller having sniffed them
  // out of InputArgs) but if they have been overridden we want the new ones.
  Invocation.getCASOpts() = Tool.getCASOpts();

  llvm::BumpPtrAllocator Alloc;
  llvm::StringSaver Saver(Alloc);
  llvm::TreePathPrefixMapper Mapper(&FS, Alloc);
  if (Error E =
          computeFullMapping(Saver, Exec, Invocation, PrefixMapping, Mapper))
    return std::move(E);

  Optional<llvm::cas::CASID> Root;
  if (Error E = Tool.getDependencyTreeFromCompilerInvocation(
                        std::make_shared<CompilerInvocation>(Invocation),
                        WorkingDirectory, DiagsConsumer,
                        [&](const llvm::vfs::CachedDirectoryEntry &Entry) {
                          return Mapper.map(Entry);
                        })
                    .moveInto(Root))
    return std::move(E);
  updateCompilerInvocation(Invocation, Saver, FS, Root->toString(),
                           WorkingDirectory, Mapper);
  return *Root;
}
