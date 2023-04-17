//===-- CacheLauncherMode.cpp - clang-cache driver mode -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CacheLauncherMode.h"
#include "clang/Basic/DiagnosticCAS.h"
#include "clang/Frontend/CompilerInvocation.h"
#include "clang/Frontend/TextDiagnosticPrinter.h"
#include "clang/Frontend/Utils.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Process.h"
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

static bool shouldCacheInvocation(ArrayRef<const char *> Args,
                                  IntrusiveRefCntPtr<DiagnosticsEngine> Diags) {
  SmallVector<const char *, 128> CheckArgs(Args.begin(), Args.end());
  // Make sure "-###" is not present otherwise we won't get an object back.
  CheckArgs.erase(
      llvm::remove_if(CheckArgs, [](StringRef Arg) { return Arg == "-###"; }),
      CheckArgs.end());
  CreateInvocationOptions Opts;
  Opts.Diags = Diags;
  // This enables picking the first invocation in a multi-arch build.
  Opts.RecoverOnError = true;
  std::shared_ptr<CompilerInvocation> CInvok =
      createInvocation(CheckArgs, std::move(Opts));
  if (!CInvok)
    return false;
  if (CInvok->getLangOpts()->Modules) {
    Diags->Report(diag::warn_clang_cache_disabled_caching)
        << "-fmodules is enabled";
    return false;
  }
  if (CInvok->getLangOpts()->AsmPreprocessor) {
    Diags->Report(diag::warn_clang_cache_disabled_caching)
        << "assembler language mode is enabled";
    return false;
  }
  if (llvm::sys::Process::GetEnv("AS_SECURE_LOG_FILE")) {
    // AS_SECURE_LOG_FILE causes uncaptured output in MC assembler.
    Diags->Report(diag::warn_clang_cache_disabled_caching)
        << "AS_SECURE_LOG_FILE is set";
    return false;
  }
  return true;
}

static int executeAsProcess(ArrayRef<const char *> Args,
                            DiagnosticsEngine &Diags) {
  SmallVector<StringRef, 128> RefArgs;
  RefArgs.reserve(Args.size());
  for (const char *Arg : Args) {
    RefArgs.push_back(Arg);
  }
  std::string ErrMsg;
  int Result =
      llvm::sys::ExecuteAndWait(RefArgs[0], RefArgs, /*Env*/ std::nullopt,
                                /*Redirects*/ {}, /*SecondsToWait*/ 0,
                                /*MemoryLimit*/ 0, &ErrMsg);
  if (!ErrMsg.empty()) {
    Diags.Report(diag::err_clang_cache_failed_execution) << ErrMsg;
  }
  return Result;
}

/// Arguments common to both \p clang-cache compiler launcher and depscan server
/// functionalities.
static void addCommonArgs(bool ForDriver, SmallVectorImpl<const char *> &Args,
                          llvm::StringSaver &Saver) {
  if (!llvm::sys::Process::GetEnv("CLANG_CACHE_USE_CASFS_DEPSCAN")) {
    Args.push_back("-fdepscan-include-tree");
  }
  auto addCC1Args = [&](ArrayRef<const char *> NewArgs) {
    for (const char *Arg : NewArgs) {
      if (ForDriver) {
        Args.append({"-Xclang", Arg});
      } else {
        Args.push_back(Arg);
      }
    }
  };
  if (auto CASPath = llvm::sys::Process::GetEnv("LLVM_CACHE_CAS_PATH")) {
    addCC1Args({"-fcas-path", Saver.save(*CASPath).data()});
  }
  if (auto PluginPath = llvm::sys::Process::GetEnv("LLVM_CACHE_PLUGIN_PATH")) {
    addCC1Args({"-fcas-plugin-path", Saver.save(*PluginPath).data()});
  }
  if (auto PluginOpts =
          llvm::sys::Process::GetEnv("LLVM_CACHE_PLUGIN_OPTIONS")) {
    StringRef Remaining = *PluginOpts;
    while (!Remaining.empty()) {
      StringRef Opt;
      std::tie(Opt, Remaining) = Remaining.split(':');
      addCC1Args({"-fcas-plugin-option", Saver.save(Opt).data()});
    }
  }
}

/// Arguments specific to \p clang-cache compiler launcher functionality.
static void addLauncherArgs(SmallVectorImpl<const char *> &Args,
                            llvm::StringSaver &Saver) {
  if (const char *DaemonPath =
          ::getenv("CLANG_CACHE_SCAN_DAEMON_SOCKET_PATH")) {
    // Instruct clang to connect to scanning daemon that is listening on the
    // provided socket path. The caller is responsible for ensuring the daemon
    // is compatible with the invoked clang.
    Args.push_back("-fdepscan=daemon");
    Args.push_back(Saver.save(Twine("-fdepscan-daemon=") + DaemonPath).data());
  } else if (const char *SessionId = ::getenv("LLVM_CACHE_BUILD_SESSION_ID")) {
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
    Args.append({"-fdepscan=daemon", "-fdepscan-share-identifier", SessionId});
  } else {
    Args.push_back("-fdepscan");
  }

  addCommonArgs(/*ForDriver*/ true, Args, Saver);

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
  if (const char *ServicePath =
          ::getenv("LLVM_CACHE_REMOTE_SERVICE_SOCKET_PATH")) {
    Args.append({"-Xclang", "-fcompilation-caching-service-path", "-Xclang",
                 ServicePath});
  }
  Args.append({"-greproducible"});
}

static void addScanServerArgs(const char *SocketPath,
                              SmallVectorImpl<const char *> &Args,
                              llvm::StringSaver &Saver) {
  Args.append({"-cc1depscand", "-serve", SocketPath, "-cas-args"});
  addCommonArgs(/*ForDriver*/ false, Args, Saver);
}

std::optional<int>
clang::handleClangCacheInvocation(SmallVectorImpl<const char *> &Args,
                                  llvm::StringSaver &Saver) {
  assert(Args.size() >= 1);

  IntrusiveRefCntPtr<DiagnosticOptions> DiagOpts;
  if (std::optional<std::string> WarnOptsValue =
          llvm::sys::Process::GetEnv("LLVM_CACHE_WARNINGS")) {
    SmallVector<const char *, 8> WarnOpts;
    WarnOpts.push_back(Args.front());
    llvm::cl::TokenizeGNUCommandLine(*WarnOptsValue, Saver, WarnOpts);
    DiagOpts = CreateAndPopulateDiagOpts(WarnOpts);
  } else {
    DiagOpts = new DiagnosticOptions();
  }
  auto DiagsConsumer = std::make_unique<TextDiagnosticPrinter>(
      llvm::errs(), DiagOpts.get(), false);
  IntrusiveRefCntPtr<DiagnosticsEngine> DiagsPtr(
      new DiagnosticsEngine(new DiagnosticIDs(), DiagOpts));
  DiagnosticsEngine &Diags = *DiagsPtr;
  Diags.setClient(DiagsConsumer.get(), /*ShouldOwnClient=*/false);
  ProcessWarningOptions(Diags, *DiagOpts);
  if (Diags.hasErrorOccurred())
    return 1;

  if (Args.size() == 1) {
    // FIXME: With just 'clang-cache' invocation consider outputting info, like
    // the on-disk CAS path and its size.
    Diags.Report(diag::err_clang_cache_missing_compiler_command);
    return 1;
  }

  const char *clangCachePath = Args.front();
  // Drop initial '/path/to/clang-cache' program name.
  Args.erase(Args.begin());

  if (StringRef(Args.front()) == "-depscan-server") {
    // Run "clang -cc1depscand -serve ..." after translating some of the
    // environment variables that \p clang-cache understands.
    if (Args.size() == 1) {
      Diags.Report(diag::err_clang_cache_scanserve_missing_args);
      return 1;
    }
    const char *SocketPath = Args[1];
    Args.clear();
    Args.push_back(clangCachePath);
    addScanServerArgs(SocketPath, Args, Saver);
    return std::nullopt;
  }

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
    if (!shouldCacheInvocation(Args, DiagsPtr)) {
      if (Diags.hasErrorOccurred())
        return 1;
      return std::nullopt;
    }
    addLauncherArgs(Args, Saver);
    return std::nullopt;
  }

  // FIXME: If it's invoking a different clang binary determine whether that
  // clang supports the caching options, don't immediately give up on caching.

  // Not invoking same clang binary, do a normal invocation without changing
  // arguments, but warn because this may be unexpected to the user.
  Diags.Report(diag::warn_clang_cache_disabled_caching)
      << "clang-cache invokes a different clang binary than itself";

  return executeAsProcess(Args, Diags);
}
