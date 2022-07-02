//===-- CacheLauncherMode.cpp - clang-cache driver mode -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CacheLauncherMode.h"
#include "clang/Basic/DiagnosticCAS.h"
#include "clang/Frontend/TextDiagnosticPrinter.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/StringSaver.h"

using namespace clang;

static bool isSameProgram(StringRef clangCachePath, StringRef compilerPath) {
  // Fast path check, see if they have the same parent path.
  if (llvm::sys::path::parent_path(clangCachePath) ==
      llvm::sys::path::parent_path(compilerPath))
    return true;
  // Check the file status IDs;
  llvm::sys::fs::file_status CacheStat, CompilerStat;
  if (llvm::sys::fs::status(clangCachePath, CacheStat))
    return false;
  if (llvm::sys::fs::status(compilerPath, CompilerStat))
    return false;
  return CacheStat.getUniqueID() == CompilerStat.getUniqueID();
}

Optional<int>
clang::handleClangCacheInvocation(SmallVectorImpl<const char *> &Args,
                                  llvm::StringSaver &Saver) {
  assert(Args.size() >= 1);

  auto DiagsConsumer = std::make_unique<TextDiagnosticPrinter>(
      llvm::errs(), new DiagnosticOptions(), false);
  DiagnosticsEngine Diags(new DiagnosticIDs(), new DiagnosticOptions());
  Diags.setClient(DiagsConsumer.get(), /*ShouldOwnClient=*/false);

  if (Args.size() == 1) {
    // FIXME: With just 'clang-cache' invocation consider outputting info, like
    // the on-disk CAS path and its size.
    Diags.Report(diag::err_clang_cache_missing_compiler_command);
    return 1;
  }

  const char *clangCachePath = Args.front();
  // Drop initial '/path/to/clang-cache' program name.
  Args.erase(Args.begin());

  llvm::ErrorOr<std::string> compilerPathOrErr =
      llvm::sys::findProgramByName(Args.front());
  if (!compilerPathOrErr) {
    Diags.Report(diag::err_clang_cache_cannot_find_binary) << Args.front();
    return 1;
  }
  std::string compilerPath = std::move(*compilerPathOrErr);
  if (Args.front() != compilerPath)
    Args[0] = Saver.save(compilerPath).data();

  if (isSameProgram(clangCachePath, compilerPath)) {
    if (const char *SessionId = ::getenv("LLVM_CACHE_BUILD_SESSION_ID")) {
      // `LLVM_CACHE_BUILD_SESSION_ID` enables sharing of a depscan daemon
      // using the string it is set to. The clang invocations under the same
      // `LLVM_CACHE_BUILD_SESSION_ID` will launch and re-use the same daemon.
      //
      // This is a scheme where we are still launching daemons on-demand,
      // instead of a scheme where we start a daemon at the beginning of the
      // "build session" for all clang invocations to connect to.
      // Launcing daemons on-demand is preferable because it allows having mixed
      // toolchains, with different clang versions, running under the same
      // `LLVM_CACHE_BUILD_SESSION_ID`; in such a case there will be one daemon
      // started and shared for each unique clang version.
      Args.append(
          {"-fdepscan=daemon", "-fdepscan-share-identifier", SessionId});
    } else {
      Args.push_back("-fdepscan");
    }
    if (const char *PrefixMaps = ::getenv("LLVM_CACHE_PREFIX_MAPS")) {
      Args.append({"-fdepscan-prefix-map-sdk=/^sdk",
                   "-fdepscan-prefix-map-toolchain=/^toolchain"});
      StringRef PrefixMap, Remaining = PrefixMaps;
      while (true) {
        std::tie(PrefixMap, Remaining) = Remaining.split(';');
        if (PrefixMap.empty())
          break;
        Args.push_back(Saver.save("-fdepscan-prefix-map=" + PrefixMap).data());
      }
    }
    if (const char *CASPath = ::getenv("LLVM_CACHE_CAS_PATH")) {
      Args.append({"-Xclang", "-fcas-path", "-Xclang", CASPath});
    }
    Args.append({"-greproducible"});
    return None;
  }

  // FIXME: If it's invoking a different clang binary determine whether that
  // clang supports the caching options, don't immediately give up on caching.

  // Not invoking same clang binary, do a normal invocation without changing
  // arguments, but warn because this may be unexpected to the user.
  Diags.Report(diag::warn_clang_cache_disabled_caching);

  SmallVector<StringRef, 128> RefArgs;
  RefArgs.reserve(Args.size());
  for (const char *Arg : Args) {
    RefArgs.push_back(Arg);
  }
  std::string ErrMsg;
  int Result = llvm::sys::ExecuteAndWait(compilerPath, RefArgs, /*Env*/ None,
                                         /*Redirects*/ {}, /*SecondsToWait*/ 0,
                                         /*MemoryLimit*/ 0, &ErrMsg);
  if (!ErrMsg.empty()) {
    Diags.Report(diag::err_clang_cache_failed_execution) << ErrMsg;
  }
  return Result;
}
