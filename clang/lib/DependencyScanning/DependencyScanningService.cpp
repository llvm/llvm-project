//===- DependencyScanningService.cpp - Scanning Service -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/DependencyScanning/DependencyScanningService.h"
#include "clang/Basic/BitmaskEnum.h"
#include "llvm/CAS/ActionCache.h"
#include "llvm/CAS/ObjectStore.h"
#include "llvm/Support/Process.h"

#include "llvm/Support/Chrono.h"

using namespace clang;
using namespace dependencies;

bool clang::dependencies::shouldCacheNegativeStatsDefault() {
  if (std::optional<std::string> MaybeNegStats =
          llvm::sys::Process::GetEnv("CLANG_SCAN_CACHE_NEGATIVE_STATS")) {
    if (MaybeNegStats->empty())
      return true;
    return llvm::StringSwitch<bool>(*MaybeNegStats)
        .Case("1", true)
        .Case("0", false)
        .Default(false);
  }
  return false;
}

// rdar://148027982
// rdar://127079541
// Negative caching directories can cause build failures due to incorrectly
// configured projects.
bool dependencies::shouldCacheNegativeStatsForPath(StringRef Path) {
  StringRef Ext = llvm::sys::path::extension(Path);
  if (Ext.empty())
    return false;
  if (Ext == ".framework")
    return false;
  return true;
}

DependencyScanningServiceOptions::DependencyScanningServiceOptions()
    : BuildSessionTimestamp(
          llvm::sys::toTimeT(std::chrono::system_clock::now())) {}

DependencyScanningService::DependencyScanningService(
    DependencyScanningServiceOptions OptsArg)
    : Opts(std::move(OptsArg)) {
  // The FullIncludeTree output format completely subsumes header search and
  // VFS optimizations due to how it works. Disable these optimizations so
  // we're not doing unneeded work.
  if (Opts.Format == ScanningOutputFormat::FullIncludeTree)
    Opts.OptimizeArgs &= ~ScanningOptimizations::FullIncludeTreeIrrelevant;
}
