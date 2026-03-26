//===- DependencyScanningService.cpp - Scanning Service -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/DependencyScanning/DependencyScanningService.h"

#include "llvm/Support/Chrono.h"

using namespace clang;
using namespace dependencies;

DependencyScanningServiceOptions::DependencyScanningServiceOptions()
    : MakeVFS([] { return llvm::vfs::createPhysicalFileSystem(); }),
      BuildSessionTimestamp(
          llvm::sys::toTimeT(std::chrono::system_clock::now())) {}
