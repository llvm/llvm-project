//===- DependencyScanner.h - Module dependency discovery --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file defines the module dependency graph and dependency-scanning
/// functionality.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_DRIVER_DEPENDENCYSCANNER_H
#define LLVM_CLANG_DRIVER_DEPENDENCYSCANNER_H

#include "clang/Driver/Types.h"

namespace llvm {
namespace vfs {
class FileSystem;
} // namespace vfs
} // namespace llvm

namespace clang {
class DiagnosticsEngine;
namespace driver {
class Compilation;
} // namespace driver
} // namespace clang

namespace clang::driver::modules {

using InputTy = std::pair<types::ID, const llvm::opt::Arg *>;

using InputList = llvm::SmallVector<InputTy, 16>;

/// Checks whether the -fmodules-driver feature should be implicitly enabled.
///
/// When -fmodules-driver is no longer experimental, it should be enabled by
/// default iff both conditions are met:
/// (1) there are two or more C++ source inputs; and
/// (2) at least one input uses C++20 named modules.
bool shouldEnableModulesDriver(const InputList &Inputs,
                               llvm::vfs::FileSystem &VFS,
                               DiagnosticsEngine &Diags);

/// Appends the std and std.compat module inputs.
bool ensureNamedModuleStdLibraryInputs(clang::driver::Compilation &C,
                                       InputList &Inputs);

/// Modifies the compilations JobList to support Clang and C++20 named  module
/// imports between source files, importing Standard library modules
bool performDriverModuleBuild(clang::driver::Compilation &C,
                              clang::DiagnosticsEngine &Diags);

} // namespace clang::driver::modules

#endif
