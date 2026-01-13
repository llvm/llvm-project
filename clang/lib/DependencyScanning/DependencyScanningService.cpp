//===- DependencyScanningService.cpp - Scanning Service -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/DependencyScanning/DependencyScanningService.h"

using namespace clang;
using namespace dependencies;

DependencyScanningService::DependencyScanningService(
    ScanningMode Mode, ScanningOutputFormat Format,
    ScanningOptimizations OptimizeArgs, bool EagerLoadModules, bool TraceVFS,
    std::time_t BuildSessionTimestamp)
    : Mode(Mode), Format(Format), OptimizeArgs(OptimizeArgs),
      EagerLoadModules(EagerLoadModules), TraceVFS(TraceVFS),
      BuildSessionTimestamp(BuildSessionTimestamp) {}
