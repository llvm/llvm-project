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
#include "llvm/CAS/CachingOnDiskFileSystem.h"
#include "llvm/CAS/ObjectStore.h"

using namespace clang;
using namespace tooling;
using namespace dependencies;

DependencyScanningService::DependencyScanningService(
    ScanningMode Mode, ScanningOutputFormat Format, CASOptions CASOpts,
    std::shared_ptr<llvm::cas::ObjectStore> CAS,
    std::shared_ptr<llvm::cas::ActionCache> Cache,
    IntrusiveRefCntPtr<llvm::cas::CachingOnDiskFileSystem> SharedFS,
    ScanningOptimizations OptimizeArgs, bool EagerLoadModules, bool TraceVFS,
    std::time_t BuildSessionTimestamp)
    : Mode(Mode), Format(Format), CASOpts(std::move(CASOpts)),
      CAS(std::move(CAS)), Cache(std::move(Cache)), OptimizeArgs(OptimizeArgs),
      EagerLoadModules(EagerLoadModules), TraceVFS(TraceVFS),
      SharedFS(std::move(SharedFS)),
      BuildSessionTimestamp(BuildSessionTimestamp) {
  if (!this->SharedFS)
    SharedCache.emplace();

  // The FullIncludeTree output format completely subsumes header search and
  // VFS optimizations due to how it works. Disable these optimizations so
  // we're not doing unneeded work.
  if (Format == ScanningOutputFormat::FullIncludeTree)
    this->OptimizeArgs &= ~ScanningOptimizations::FullIncludeTreeIrrelevant;
}
