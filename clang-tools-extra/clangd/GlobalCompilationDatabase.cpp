//===--- GlobalCompilationDatabase.cpp ---------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "GlobalCompilationDatabase.h"
#include "FS.h"
#include "Logger.h"
#include "Path.h"
#include "clang/Frontend/CompilerInvocation.h"
#include "clang/Tooling/ArgumentsAdjusters.h"
#include "clang/Tooling/CompilationDatabase.h"
#include "llvm/ADT/None.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include <string>
#include <tuple>
#include <vector>

namespace clang {
namespace clangd {
namespace {

// Query apple's `xcrun` launcher, which is the source of truth for "how should"
// clang be invoked on this system.
llvm::Optional<std::string> queryXcrun(llvm::ArrayRef<llvm::StringRef> Argv) {
  auto Xcrun = llvm::sys::findProgramByName("xcrun");
  if (!Xcrun) {
    log("Couldn't find xcrun. Hopefully you have a non-apple toolchain...");
    return llvm::None;
  }
  llvm::SmallString<64> OutFile;
  llvm::sys::fs::createTemporaryFile("clangd-xcrun", "", OutFile);
  llvm::FileRemover OutRemover(OutFile);
  llvm::Optional<llvm::StringRef> Redirects[3] = {
      /*stdin=*/{""}, /*stdout=*/{OutFile}, /*stderr=*/{""}};
  vlog("Invoking {0} to find clang installation", *Xcrun);
  int Ret = llvm::sys::ExecuteAndWait(*Xcrun, Argv,
                                      /*Env=*/llvm::None, Redirects,
                                      /*SecondsToWait=*/10);
  if (Ret != 0) {
    log("xcrun exists but failed with code {0}. "
        "If you have a non-apple toolchain, this is OK. "
        "Otherwise, try xcode-select --install.",
        Ret);
    return llvm::None;
  }

  auto Buf = llvm::MemoryBuffer::getFile(OutFile);
  if (!Buf) {
    log("Can't read xcrun output: {0}", Buf.getError().message());
    return llvm::None;
  }
  StringRef Path = Buf->get()->getBuffer().trim();
  if (Path.empty()) {
    log("xcrun produced no output");
    return llvm::None;
  }
  return Path.str();
}

// On Mac, `which clang` is /usr/bin/clang. It runs `xcrun clang`, which knows
// where the real clang is kept. We need to do the same thing,
// because cc1 (not the driver!) will find libc++ relative to argv[0].
llvm::Optional<std::string> queryMacClangPath() {
#ifndef __APPLE__
  return llvm::None;
#endif

  return queryXcrun({"xcrun", "--find", "clang"});
}

// Resolve symlinks if possible.
std::string resolve(std::string Path) {
  llvm::SmallString<128> Resolved;
  if (llvm::sys::fs::real_path(Path, Resolved))
    return Path; // On error;
  return Resolved.str();
}

// Get a plausible full `clang` path.
// This is used in the fallback compile command, or when the CDB returns a
// generic driver with no path.
llvm::StringRef getFallbackClangPath() {
  static const std::string &MemoizedFallbackPath = [&]() -> std::string {
    // The driver and/or cc1 sometimes depend on the binary name to compute
    // useful things like the standard library location.
    // We need to emulate what clang on this system is likely to see.
    // cc1 in particular looks at the "real path" of the running process, and
    // so if /usr/bin/clang is a symlink, it sees the resolved path.
    // clangd doesn't have that luxury, so we resolve symlinks ourselves.

    // /usr/bin/clang on a mac is a program that redirects to the right clang.
    // We resolve it as if it were a symlink.
    if (auto MacClang = queryMacClangPath())
      return resolve(std::move(*MacClang));
    // On other platforms, just look for compilers on the PATH.
    for (const char* Name : {"clang", "gcc", "cc"})
      if (auto PathCC = llvm::sys::findProgramByName(Name))
        return resolve(std::move(*PathCC));
    // Fallback: a nonexistent 'clang' binary next to clangd.
    static int Dummy;
    std::string ClangdExecutable =
        llvm::sys::fs::getMainExecutable("clangd", (void *)&Dummy);
    SmallString<128> ClangPath;
    ClangPath = llvm::sys::path::parent_path(ClangdExecutable);
    llvm::sys::path::append(ClangPath, "clang");
    return ClangPath.str();
  }();
  return MemoizedFallbackPath;
}

// On mac, /usr/bin/clang sets SDKROOT and then invokes the real clang.
// The effect of this is to set -isysroot correctly. We do the same.
const std::string *getMacSysroot() {
#ifndef __APPLE__
  return nullptr;
#endif

  // SDKROOT overridden in environment, respect it. Driver will set isysroot.
  if (::getenv("SDKROOT"))
    return nullptr;
  static const llvm::Optional<std::string> &Sysroot =
      queryXcrun({"xcrun", "--show-sdk-path"});
  return Sysroot ? Sysroot.getPointer() : nullptr;
}

// Transform a command into the form we want to send to the driver.
// The command was originally either from the CDB or is the fallback command,
// and may have been modified by OverlayCDB.
void adjustArguments(tooling::CompileCommand &Cmd,
                     llvm::StringRef ResourceDir) {
  tooling::ArgumentsAdjuster ArgsAdjuster = tooling::combineAdjusters(
      // clangd should not write files to disk, including dependency files
      // requested on the command line.
      tooling::getClangStripDependencyFileAdjuster(),
      // Strip plugin related command line arguments. Clangd does
      // not support plugins currently. Therefore it breaks if
      // compiler tries to load plugins.
      tooling::combineAdjusters(tooling::getStripPluginsAdjuster(),
                                tooling::getClangSyntaxOnlyAdjuster()));

  Cmd.CommandLine = ArgsAdjuster(Cmd.CommandLine, Cmd.Filename);
  // Check whether the flag exists, either as -flag or -flag=*
  auto Has = [&](llvm::StringRef Flag) {
    for (llvm::StringRef Arg : Cmd.CommandLine) {
      if (Arg.consume_front(Flag) && (Arg.empty() || Arg[0] == '='))
        return true;
    }
    return false;
  };
  // Inject the resource dir.
  if (!ResourceDir.empty() && !Has("-resource-dir"))
    Cmd.CommandLine.push_back(("-resource-dir=" + ResourceDir).str());
  if (!Has("-isysroot"))
    if (const std::string *MacSysroot = getMacSysroot()) {
      Cmd.CommandLine.push_back("-isysroot");
      Cmd.CommandLine.push_back(*MacSysroot);
    }

  // If the driver is a generic name like "g++" with no path, add a clang path.
  // This makes it easier for us to find the standard libraries on mac.
  if (!Cmd.CommandLine.empty()) {
    std::string &Driver = Cmd.CommandLine.front();
    if (Driver == "clang" || Driver == "clang++" || Driver == "gcc" ||
        Driver == "g++" || Driver == "cc" || Driver == "c++") {
      llvm::SmallString<128> QualifiedDriver =
          llvm::sys::path::parent_path(getFallbackClangPath());
      llvm::sys::path::append(QualifiedDriver, Driver);
      Driver = QualifiedDriver.str();
    }
  }
}

std::string getStandardResourceDir() {
  static int Dummy; // Just an address in this process.
  return CompilerInvocation::GetResourcesPath("clangd", (void *)&Dummy);
}

// Runs the given action on all parent directories of filename, starting from
// deepest directory and going up to root. Stops whenever action succeeds.
void actOnAllParentDirectories(PathRef FileName,
                               llvm::function_ref<bool(PathRef)> Action) {
  for (auto Path = llvm::sys::path::parent_path(FileName);
       !Path.empty() && !Action(Path);
       Path = llvm::sys::path::parent_path(Path))
    ;
}

} // namespace

tooling::CompileCommand
GlobalCompilationDatabase::getFallbackCommand(PathRef File) const {
  std::vector<std::string> Argv = {"clang"};
  // Clang treats .h files as C by default and files without extension as linker
  // input, resulting in unhelpful diagnostics.
  // Parsing as Objective C++ is friendly to more cases.
  auto FileExtension = llvm::sys::path::extension(File);
  if (FileExtension.empty() || FileExtension == ".h")
    Argv.push_back("-xobjective-c++-header");
  Argv.push_back(File);
  tooling::CompileCommand Cmd(llvm::sys::path::parent_path(File),
                              llvm::sys::path::filename(File), std::move(Argv),
                              /*Output=*/"");
  Cmd.Heuristic = "clangd fallback";
  return Cmd;
}

DirectoryBasedGlobalCompilationDatabase::
    DirectoryBasedGlobalCompilationDatabase(
        llvm::Optional<Path> CompileCommandsDir)
    : CompileCommandsDir(std::move(CompileCommandsDir)) {}

DirectoryBasedGlobalCompilationDatabase::
    ~DirectoryBasedGlobalCompilationDatabase() = default;

llvm::Optional<tooling::CompileCommand>
DirectoryBasedGlobalCompilationDatabase::getCompileCommand(PathRef File) const {
  CDBLookupRequest Req;
  Req.FileName = File;
  Req.ShouldBroadcast = true;

  auto Res = lookupCDB(Req);
  if (!Res) {
    log("Failed to find compilation database for {0}", File);
    return llvm::None;
  }

  auto Candidates = Res->CDB->getCompileCommands(File);
  if (!Candidates.empty())
    return std::move(Candidates.front());

  return None;
}

// For platforms where paths are case-insensitive (but case-preserving),
// we need to do case-insensitive comparisons and use lowercase keys.
// FIXME: Make Path a real class with desired semantics instead.
//        This class is not the only place this problem exists.
// FIXME: Mac filesystems default to case-insensitive, but may be sensitive.

static std::string maybeCaseFoldPath(PathRef Path) {
#if defined(_WIN32) || defined(__APPLE__)
  return Path.lower();
#else
  return Path;
#endif
}

static bool pathEqual(PathRef A, PathRef B) {
#if defined(_WIN32) || defined(__APPLE__)
  return A.equals_lower(B);
#else
  return A == B;
#endif
}

DirectoryBasedGlobalCompilationDatabase::CachedCDB &
DirectoryBasedGlobalCompilationDatabase::getCDBInDirLocked(PathRef Dir) const {
  // FIXME(ibiryukov): Invalidate cached compilation databases on changes
  // FIXME(sammccall): this function hot, avoid copying key when hitting cache.
  auto Key = maybeCaseFoldPath(Dir);
  auto R = CompilationDatabases.try_emplace(Key);
  if (R.second) { // Cache miss, try to load CDB.
    CachedCDB &Entry = R.first->second;
    std::string Error = "";
    Entry.CDB = tooling::CompilationDatabase::loadFromDirectory(Dir, Error);
    Entry.Path = Dir;
  }
  return R.first->second;
}

llvm::Optional<DirectoryBasedGlobalCompilationDatabase::CDBLookupResult>
DirectoryBasedGlobalCompilationDatabase::lookupCDB(
    CDBLookupRequest Request) const {
  assert(llvm::sys::path::is_absolute(Request.FileName) &&
         "path must be absolute");

  bool ShouldBroadcast = false;
  CDBLookupResult Result;

  {
    std::lock_guard<std::mutex> Lock(Mutex);
    CachedCDB *Entry = nullptr;
    if (CompileCommandsDir) {
      Entry = &getCDBInDirLocked(*CompileCommandsDir);
    } else {
      // Traverse the canonical version to prevent false positives. i.e.:
      // src/build/../a.cc can detect a CDB in /src/build if not canonicalized.
      // FIXME(sammccall): this loop is hot, use a union-find-like structure.
      actOnAllParentDirectories(removeDots(Request.FileName),
                                [&](PathRef Path) {
                                  Entry = &getCDBInDirLocked(Path);
                                  return Entry->CDB != nullptr;
                                });
    }

    if (!Entry || !Entry->CDB)
      return llvm::None;

    // Mark CDB as broadcasted to make sure discovery is performed once.
    if (Request.ShouldBroadcast && !Entry->SentBroadcast) {
      Entry->SentBroadcast = true;
      ShouldBroadcast = true;
    }

    Result.CDB = Entry->CDB.get();
    Result.PI.SourceRoot = Entry->Path;
  }

  // FIXME: Maybe make the following part async, since this can block retrieval
  // of compile commands.
  if (ShouldBroadcast)
    broadcastCDB(Result);
  return Result;
}

void DirectoryBasedGlobalCompilationDatabase::broadcastCDB(
    CDBLookupResult Result) const {
  assert(Result.CDB && "Trying to broadcast an invalid CDB!");

  std::vector<std::string> AllFiles = Result.CDB->getAllFiles();
  // We assume CDB in CompileCommandsDir owns all of its entries, since we don't
  // perform any search in parent paths whenever it is set.
  if (CompileCommandsDir) {
    assert(*CompileCommandsDir == Result.PI.SourceRoot &&
           "Trying to broadcast a CDB outside of CompileCommandsDir!");
    OnCommandChanged.broadcast(std::move(AllFiles));
    return;
  }

  llvm::StringMap<bool> DirectoryHasCDB;
  {
    std::lock_guard<std::mutex> Lock(Mutex);
    // Uniquify all parent directories of all files.
    for (llvm::StringRef File : AllFiles) {
      actOnAllParentDirectories(File, [&](PathRef Path) {
        auto It = DirectoryHasCDB.try_emplace(Path);
        // Already seen this path, and all of its parents.
        if (!It.second)
          return true;

        CachedCDB &Entry = getCDBInDirLocked(Path);
        It.first->second = Entry.CDB != nullptr;
        return pathEqual(Path, Result.PI.SourceRoot);
      });
    }
  }

  std::vector<std::string> GovernedFiles;
  for (llvm::StringRef File : AllFiles) {
    // A file is governed by this CDB if lookup for the file would find it.
    // Independent of whether it has an entry for that file or not.
    actOnAllParentDirectories(File, [&](PathRef Path) {
      if (DirectoryHasCDB.lookup(Path)) {
        if (pathEqual(Path, Result.PI.SourceRoot))
          // Make sure listeners always get a canonical path for the file.
          GovernedFiles.push_back(removeDots(File));
        // Stop as soon as we hit a CDB.
        return true;
      }
      return false;
    });
  }

  OnCommandChanged.broadcast(std::move(GovernedFiles));
}

llvm::Optional<ProjectInfo>
DirectoryBasedGlobalCompilationDatabase::getProjectInfo(PathRef File) const {
  CDBLookupRequest Req;
  Req.FileName = File;
  Req.ShouldBroadcast = false;
  auto Res = lookupCDB(Req);
  if (!Res)
    return llvm::None;
  return Res->PI;
}

OverlayCDB::OverlayCDB(const GlobalCompilationDatabase *Base,
                       std::vector<std::string> FallbackFlags,
                       llvm::Optional<std::string> ResourceDir)
    : Base(Base), ResourceDir(ResourceDir ? std::move(*ResourceDir)
                                          : getStandardResourceDir()),
      FallbackFlags(std::move(FallbackFlags)) {
  if (Base)
    BaseChanged = Base->watch([this](const std::vector<std::string> Changes) {
      OnCommandChanged.broadcast(Changes);
    });
}

llvm::Optional<tooling::CompileCommand>
OverlayCDB::getCompileCommand(PathRef File) const {
  llvm::Optional<tooling::CompileCommand> Cmd;
  {
    std::lock_guard<std::mutex> Lock(Mutex);
    auto It = Commands.find(removeDots(File));
    if (It != Commands.end())
      Cmd = It->second;
  }
  if (!Cmd && Base)
    Cmd = Base->getCompileCommand(File);
  if (!Cmd)
    return llvm::None;
  adjustArguments(*Cmd, ResourceDir);
  return Cmd;
}

tooling::CompileCommand OverlayCDB::getFallbackCommand(PathRef File) const {
  auto Cmd = Base ? Base->getFallbackCommand(File)
                  : GlobalCompilationDatabase::getFallbackCommand(File);
  std::lock_guard<std::mutex> Lock(Mutex);
  Cmd.CommandLine.insert(Cmd.CommandLine.end(), FallbackFlags.begin(),
                         FallbackFlags.end());
  adjustArguments(Cmd, ResourceDir);
  return Cmd;
}

void OverlayCDB::setCompileCommand(
    PathRef File, llvm::Optional<tooling::CompileCommand> Cmd) {
  // We store a canonical version internally to prevent mismatches between set
  // and get compile commands. Also it assures clients listening to broadcasts
  // doesn't receive different names for the same file.
  std::string CanonPath = removeDots(File);
  {
    std::unique_lock<std::mutex> Lock(Mutex);
    if (Cmd)
      Commands[CanonPath] = std::move(*Cmd);
    else
      Commands.erase(CanonPath);
  }
  OnCommandChanged.broadcast({CanonPath});
}

llvm::Optional<ProjectInfo> OverlayCDB::getProjectInfo(PathRef File) const {
  {
    std::lock_guard<std::mutex> Lock(Mutex);
    auto It = Commands.find(removeDots(File));
    if (It != Commands.end())
      return ProjectInfo{};
  }
  if (Base)
    return Base->getProjectInfo(File);

  return llvm::None;
}
} // namespace clangd
} // namespace clang
