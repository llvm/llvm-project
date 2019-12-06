//===--- CompileCommands.cpp ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CompileCommands.h"
#include "Logger.h"
#include "clang/Frontend/CompilerInvocation.h"
#include "clang/Tooling/ArgumentsAdjusters.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"

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

// Resolve symlinks if possible.
std::string resolve(std::string Path) {
  llvm::SmallString<128> Resolved;
  if (llvm::sys::fs::real_path(Path, Resolved)) {
    log("Failed to resolve possible symlink {0}", Path);
    return Path;
  }
  return Resolved.str();
}

// Get a plausible full `clang` path.
// This is used in the fallback compile command, or when the CDB returns a
// generic driver with no path.
std::string detectClangPath() {
  // The driver and/or cc1 sometimes depend on the binary name to compute
  // useful things like the standard library location.
  // We need to emulate what clang on this system is likely to see.
  // cc1 in particular looks at the "real path" of the running process, and
  // so if /usr/bin/clang is a symlink, it sees the resolved path.
  // clangd doesn't have that luxury, so we resolve symlinks ourselves.

  // On Mac, `which clang` is /usr/bin/clang. It runs `xcrun clang`, which knows
  // where the real clang is kept. We need to do the same thing,
  // because cc1 (not the driver!) will find libc++ relative to argv[0].
#ifdef __APPLE__
  if (auto MacClang = queryXcrun({"xcrun", "--find", "clang"}))
    return resolve(std::move(*MacClang));
#endif
  // On other platforms, just look for compilers on the PATH.
  for (const char *Name : {"clang", "gcc", "cc"})
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
}

// On mac, /usr/bin/clang sets SDKROOT and then invokes the real clang.
// The effect of this is to set -isysroot correctly. We do the same.
const llvm::Optional<std::string> detectSysroot() {
#ifndef __APPLE__
  return llvm::None;
#endif

  // SDKROOT overridden in environment, respect it. Driver will set isysroot.
  if (::getenv("SDKROOT"))
    return llvm::None;
  return queryXcrun({"xcrun", "--show-sdk-path"});
  return llvm::None;
}

std::string detectStandardResourceDir() {
  static int Dummy; // Just an address in this process.
  return CompilerInvocation::GetResourcesPath("clangd", (void *)&Dummy);
}

} // namespace

CommandMangler CommandMangler::detect() {
  CommandMangler Result;
  Result.ClangPath = detectClangPath();
  Result.ResourceDir = detectStandardResourceDir();
  Result.Sysroot = detectSysroot();
  return Result;
}

CommandMangler CommandMangler::forTests() {
  return CommandMangler();
}

void CommandMangler::adjust(std::vector<std::string> &Cmd) const {
  // Check whether the flag exists, either as -flag or -flag=*
  auto Has = [&](llvm::StringRef Flag) {
    for (llvm::StringRef Arg : Cmd) {
      if (Arg.consume_front(Flag) && (Arg.empty() || Arg[0] == '='))
        return true;
    }
    return false;
  };

  // clangd should not write files to disk, including dependency files
  // requested on the command line.
  Cmd = tooling::getClangStripDependencyFileAdjuster()(Cmd, "");
  // Strip plugin related command line arguments. Clangd does
  // not support plugins currently. Therefore it breaks if
  // compiler tries to load plugins.
  Cmd = tooling::getStripPluginsAdjuster()(Cmd, "");
  Cmd = tooling::getClangSyntaxOnlyAdjuster()(Cmd, "");

  if (ResourceDir && !Has("-resource-dir"))
    Cmd.push_back(("-resource-dir=" + *ResourceDir));

  if (Sysroot && !Has("-isysroot")) {
    Cmd.push_back("-isysroot");
    Cmd.push_back(*Sysroot);
  }

  // If the driver is a generic name like "g++" with no path, add a clang path.
  // This makes it easier for us to find the standard libraries on mac.
  if (ClangPath && llvm::sys::path::is_absolute(*ClangPath) && !Cmd.empty()) {
    std::string &Driver = Cmd.front();
    if (Driver == "clang" || Driver == "clang++" || Driver == "gcc" ||
        Driver == "g++" || Driver == "cc" || Driver == "c++") {
      llvm::SmallString<128> QualifiedDriver =
          llvm::sys::path::parent_path(*ClangPath);
      llvm::sys::path::append(QualifiedDriver, Driver);
      Driver = QualifiedDriver.str();
    }
  }
}

CommandMangler::operator clang::tooling::ArgumentsAdjuster() {
  return [Mangler{*this}](const std::vector<std::string> &Args,
                          llvm::StringRef File) {
    auto Result = Args;
    Mangler.adjust(Result);
    return Result;
  };
}

} // namespace clangd
} // namespace clang
