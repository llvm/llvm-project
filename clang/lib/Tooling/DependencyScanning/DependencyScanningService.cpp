//===- DependencyScanningService.cpp - clang-scan-deps service ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Tooling/DependencyScanning/DependencyScanningService.h"
#include "llvm/CAS/CASDB.h"
#include "llvm/CAS/CachingOnDiskFileSystem.h"
#include "llvm/Support/TargetSelect.h"

using namespace clang;
using namespace tooling;
using namespace dependencies;

DependencyScanningService::DependencyScanningService(
    ScanningMode Mode, ScanningOutputFormat Format, CASOptions CASOpts,
    IntrusiveRefCntPtr<llvm::cas::CachingOnDiskFileSystem> SharedFS,
    bool ReuseFileManager, bool OptimizeArgs, bool EagerLoadModules)
    : Mode(Mode), Format(Format), CASOpts(std::move(CASOpts)),
      ReuseFileManager(ReuseFileManager), OptimizeArgs(OptimizeArgs),
      SharedFS(std::move(SharedFS)), EagerLoadModules(EagerLoadModules) {
  if (!this->SharedFS)
    SharedCache.emplace();

  // Initialize targets for object file support.
  llvm::InitializeAllTargets();
  llvm::InitializeAllTargetMCs();
  llvm::InitializeAllAsmPrinters();
  llvm::InitializeAllAsmParsers();
}
