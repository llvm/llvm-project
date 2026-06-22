//===- DependencyScanningService.cpp - Scanning Service -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/DependencyScanning/DependencyScanningService.h"

#include "llvm/Support/Chrono.h"
#include "llvm/Support/Process.h"

using namespace clang;
using namespace dependencies;

bool dependencies::shouldCacheNegativeStatsDefault() {
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
    : MakeVFS([] { return llvm::vfs::createPhysicalFileSystem(); }),
      BuildSessionTimestamp(
          llvm::sys::toTimeT(std::chrono::system_clock::now())) {}
