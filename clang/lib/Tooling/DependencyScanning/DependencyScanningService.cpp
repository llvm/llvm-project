//===- DependencyScanningService.cpp - clang-scan-deps service ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Tooling/DependencyScanning/DependencyScanningService.h"
#include "clang/Basic/BitmaskEnum.h"
#include "llvm/CAS/ActionCache.h"
#include "llvm/CAS/ObjectStore.h"
#include "llvm/Support/Process.h"

using namespace clang;
using namespace tooling;
using namespace dependencies;

bool clang::tooling::dependencies::shouldCacheNegativeStatsDefault() {
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

DependencyScanningService::DependencyScanningService(
    ScanningMode Mode, ScanningOutputFormat Format, CASOptions CASOpts,
    std::shared_ptr<llvm::cas::ObjectStore> CAS,
    std::shared_ptr<llvm::cas::ActionCache> Cache,
    ScanningOptimizations OptimizeArgs, bool EagerLoadModules, bool TraceVFS,
    std::time_t BuildSessionTimestamp, bool CacheNegativeStats)
    : Mode(Mode), Format(Format), CASOpts(std::move(CASOpts)),
      CAS(std::move(CAS)), Cache(std::move(Cache)), OptimizeArgs(OptimizeArgs),
      EagerLoadModules(EagerLoadModules), TraceVFS(TraceVFS),
      CacheNegativeStats(CacheNegativeStats),
      BuildSessionTimestamp(BuildSessionTimestamp) {
  // The FullIncludeTree output format completely subsumes header search and
  // VFS optimizations due to how it works. Disable these optimizations so
  // we're not doing unneeded work.
  if (Format == ScanningOutputFormat::FullIncludeTree)
    this->OptimizeArgs &= ~ScanningOptimizations::FullIncludeTreeIrrelevant;
}
