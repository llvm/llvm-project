//===- ModulesDriver.h - Driver managed module builds --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file defines functionality to support driver managed builds for
/// compilations which use Clang modules or standard C++20 named modules.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_DRIVER_MODULESDRIVER_H
#define LLVM_CLANG_DRIVER_MODULESDRIVER_H

#include "clang/Driver/Types.h"
#include "llvm/Support/Error.h"

namespace llvm {
namespace vfs {
class FileSystem;
} // namespace vfs
namespace opt {
class Arg;
} // namespace opt
} // namespace llvm

namespace clang {
class DiagnosticsEngine;
namespace driver {
class Compilation;
} // namespace driver
} // namespace clang

namespace clang::driver::modules {

/// A list of inputs and their types for the given arguments.
/// Identical to Driver::InputTy.
using InputTy = std::pair<types::ID, const llvm::opt::Arg *>;

/// A list of inputs and their types for the given arguments.
/// Identical to Driver::InputList.
using InputList = llvm::SmallVector<InputTy, 16>;

/// Checks whether the -fmodules-driver feature should be implicitly enabled.
///
/// The modules driver should be enabled if both:
/// 1) the compilation has more than two C++ source inputs; and
/// 2) any C++ source inputs uses C++20 named modules.
///
/// \param Inputs The input list for the compilation being built.
/// \param VFS The virtual file system to use for all reads.
/// \param Diags The diagnostics engine used for emitting remarks only.
///
/// \returns True if the modules driver should be enabled, false otherwise,
/// or a llvm::FileError on failure to read a source input.
llvm::Expected<bool> shouldUseModulesDriver(const InputList &Inputs,
                                            llvm::vfs::FileSystem &VFS,
                                            DiagnosticsEngine &Diags);

/// The parsed Standard library module manifest.
struct StdModuleManifest {
  struct LocalModuleArgs {
    std::vector<std::string> SystemIncludeDirs;
  };

  struct Module {
    bool IsStdlib = false;
    std::string LogicalName;
    std::string SourcePath;
    std::optional<LocalModuleArgs> LocalArgs;
  };

  std::vector<Module> ModuleEntries;
};

/// Reads the Standard library module manifest from the specified path.
///
/// All source file paths in the returned manifest are made absolute.
///
/// \param StdModuleManifestPath Path to the manifest file.
/// \param VFS The llvm::vfs::FileSystem to be used for all file accesses.
///
/// \returns The parsed manifest on success, a llvm::FileError on failure to
/// read the manifest, or a llvm::json::ParseError on failure to parse it.
llvm::Expected<StdModuleManifest>
readStdModuleManifest(llvm::StringRef StdModuleManifestPath,
                      llvm::vfs::FileSystem &VFS);

/// Constructs compilation inputs for each module listed in the provided
/// Standard library module manifest.
///
/// \param Manifest The standard modules manifest
/// \param C The compilation being built.
/// \param Inputs The input list to which the corresponding input entries are
/// appended.
void buildStdModuleManifestInputs(const StdModuleManifest &Manifest,
                                  Compilation &C, InputList &Inputs);

/// Reorders and builds compilation jobs based on discovered module
/// dependencies.
///
/// \param C The compilation being built.
/// \param Manifest The standard modules manifest
void planDriverManagedModuleCompilation(Compilation &C,
                                        const StdModuleManifest &Manifest);

} // namespace clang::driver::modules

#endif
