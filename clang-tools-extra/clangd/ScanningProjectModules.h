//===------------ ScanningProjectModules.h -----------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANGD_SCANNINGPROJECTMODULES_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANGD_SCANNINGPROJECTMODULES_H

#include "ProjectModules.h"
#include "clang/Tooling/CompilationDatabase.h"

namespace clang {
namespace clangd {

/// Providing modules information for the project by scanning every file.
std::unique_ptr<ProjectModules> scanningProjectModules(
    std::shared_ptr<const clang::tooling::CompilationDatabase> CDB,
    const ThreadsafeFS &TFS);

} // namespace clangd
} // namespace clang

#endif
