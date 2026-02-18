//===--- ModulesDriver.cpp - Driver managed module builds -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file defines functionality to support driver managed builds for
/// compilations which use Clang modules or standard C++20 named modules.
///
//===----------------------------------------------------------------------===//

#include "clang/Driver/ModulesDriver.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/LLVM.h"
#include "clang/Driver/Compilation.h"
#include "clang/Driver/Driver.h"
#include "clang/Driver/Job.h"
#include "clang/Driver/Tool.h"
#include "clang/Driver/ToolChain.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/VirtualFileSystem.h"

using namespace llvm::opt;
using namespace clang;
using namespace driver;
using namespace modules;

namespace clang::driver::modules {
static bool fromJSON(const llvm::json::Value &Params,
                     StdModuleManifest::Module::LocalArguments &LocalArgs,
                     llvm::json::Path P) {
  llvm::json::ObjectMapper O(Params, P);
  return O.mapOptional("system-include-directories",
                       LocalArgs.SystemIncludeDirs);
}

static bool fromJSON(const llvm::json::Value &Params,
                     StdModuleManifest::Module &ModuleEntry,
                     llvm::json::Path P) {
  llvm::json::ObjectMapper O(Params, P);
  return O.map("is-std-library", ModuleEntry.IsStdlib) &&
         O.map("logical-name", ModuleEntry.LogicalName) &&
         O.map("source-path", ModuleEntry.SourcePath) &&
         O.mapOptional("local-arguments", ModuleEntry.LocalArgs);
}

static bool fromJSON(const llvm::json::Value &Params,
                     StdModuleManifest &Manifest, llvm::json::Path P) {
  llvm::json::ObjectMapper O(Params, P);
  return O.map("modules", Manifest.Modules);
}
} // namespace clang::driver::modules

/// Parses the Standard library module manifest from \p Buffer.
static Expected<StdModuleManifest> parseManifest(StringRef Buffer) {
  auto ParsedOrErr = llvm::json::parse(Buffer);
  if (!ParsedOrErr)
    return ParsedOrErr.takeError();

  StdModuleManifest Manifest;
  llvm::json::Path::Root Root;
  if (!fromJSON(*ParsedOrErr, Manifest, Root))
    return Root.getError();

  return Manifest;
}

/// Converts each file path in manifest from relative to absolute.
///
/// Each file path in the manifest is expected to be relative the manifest's
/// location \p ManifestPath itself.
static void makeManifestPathsAbsolute(
    MutableArrayRef<StdModuleManifest::Module> ManifestEntries,
    StringRef ManifestPath) {
  StringRef ManifestDir = llvm::sys::path::parent_path(ManifestPath);
  SmallString<256> TempPath;

  auto PrependManifestDir = [&](StringRef Path) {
    TempPath = ManifestDir;
    llvm::sys::path::append(TempPath, Path);
    return std::string(TempPath);
  };

  for (auto &Entry : ManifestEntries) {
    Entry.SourcePath = PrependManifestDir(Entry.SourcePath);
    if (!Entry.LocalArgs)
      continue;

    for (auto &IncludeDir : Entry.LocalArgs->SystemIncludeDirs)
      IncludeDir = PrependManifestDir(IncludeDir);
  }
}

Expected<StdModuleManifest>
driver::modules::readStdModuleManifest(StringRef ManifestPath,
                                       llvm::vfs::FileSystem &VFS) {
  auto MemBufOrErr = VFS.getBufferForFile(ManifestPath);
  if (!MemBufOrErr)
    return llvm::createFileError(ManifestPath, MemBufOrErr.getError());

  auto ManifestOrErr = parseManifest((*MemBufOrErr)->getBuffer());
  if (!ManifestOrErr)
    return ManifestOrErr.takeError();
  auto Manifest = std::move(*ManifestOrErr);

  makeManifestPathsAbsolute(Manifest.Modules, ManifestPath);
  return Manifest;
}

void driver::modules::buildStdModuleManifestInputs(
    ArrayRef<StdModuleManifest::Module> ManifestEntries, Compilation &C,
    InputList &Inputs) {
  DerivedArgList &Args = C.getArgs();
  const OptTable &Opts = C.getDriver().getOpts();
  for (const auto &Entry : ManifestEntries) {
    auto *InputArg =
        makeInputArg(Args, Opts, Args.MakeArgString(Entry.SourcePath));
    Inputs.emplace_back(types::TY_CXXModule, InputArg);
  }
}

using ManifestEntryLookup =
    llvm::DenseMap<StringRef, const StdModuleManifest::Module *>;

/// Builds a mapping from a module's source path to its entry in the manifest.
static ManifestEntryLookup
buildManifestLookupMap(ArrayRef<StdModuleManifest::Module> ManifestEntries) {
  ManifestEntryLookup ManifestEntryBySource;
  for (auto &Entry : ManifestEntries) {
    const bool Inserted =
        ManifestEntryBySource.try_emplace(Entry.SourcePath, &Entry).second;
    assert(Inserted &&
           "Manifest defines multiple modules with the same source path.");
  }
  return ManifestEntryBySource;
}

/// Returns the manifest entry corresponding to \p Job, or \c nullptr if none
/// exists.
static const StdModuleManifest::Module *
getManifestEntryForCommand(const Command &Job,
                           const ManifestEntryLookup &ManifestEntryBySource) {
  for (const auto &II : Job.getInputInfos()) {
    if (const auto It = ManifestEntryBySource.find(II.getFilename());
        It != ManifestEntryBySource.end())
      return It->second;
  }
  return nullptr;
}

/// Adds all \p SystemIncludeDirs to \p Job's arguments.
static void
addSystemIncludeDirsFromManifest(Compilation &C, Command &Job,
                                 ArrayRef<std::string> SystemIncludeDirs) {
  const ToolChain &TC = Job.getCreator().getToolChain();
  const DerivedArgList &TCArgs =
      C.getArgsForToolChain(&TC, Job.getSource().getOffloadingArch(),
                            Job.getSource().getOffloadingDeviceKind());

  ArgStringList NewArgs = Job.getArguments();
  for (const auto &IncludeDir : SystemIncludeDirs)
    TC.addSystemInclude(TCArgs, NewArgs, IncludeDir);
  Job.replaceArguments(NewArgs);
}

static bool isCC1Job(const Command &Job) {
  return StringRef(Job.getCreator().getName()) == "clang";
}

/// For each job that generates a Standard library module, applies any
/// local arguments specified in the corresponding module manifest entry.
static void
applyLocalArgsFromManifest(Compilation &C,
                           const ManifestEntryLookup &ManifestEntryBySource,
                           MutableArrayRef<std::unique_ptr<Command>> Jobs) {
  for (auto &Job : Jobs) {
    if (!isCC1Job(*Job))
      continue;

    const auto *Entry = getManifestEntryForCommand(*Job, ManifestEntryBySource);
    if (!Entry || !Entry->LocalArgs)
      continue;

    const auto &IncludeDirs = Entry->LocalArgs->SystemIncludeDirs;
    addSystemIncludeDirsFromManifest(C, *Job, IncludeDirs);
  }
}

void driver::modules::runModulesDriver(
    Compilation &C, ArrayRef<StdModuleManifest::Module> ManifestEntries) {
  llvm::PrettyStackTraceString CrashInfo("Running modules driver.");

  auto Jobs = C.getJobs().takeJobs();
  assert(ManifestEntries.size() <= Jobs.size() &&
         "Expected at least one job per Standard library module input");

  // Apply manifest-specified local arguments before scanning as they may affect
  // the scan results.
  const auto ManifestEntryBySource = buildManifestLookupMap(ManifestEntries);
  applyLocalArgsFromManifest(C, ManifestEntryBySource, Jobs);

  // TODO: Run the dependency scan.
  // TODO: Reorder and modify jobs based on the discovered dependencies.
}
