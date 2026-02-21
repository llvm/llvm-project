//===----------------------------------------------------------------------===//
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

#include "clang/Basic/LLVM.h"
#include "clang/Driver/Types.h"
#include "llvm/Support/Error.h"

namespace llvm::vfs {
class FileSystem;
} // namespace llvm::vfs

namespace clang {
class DiagnosticsEngine;
namespace driver {
class Compilation;
} // namespace driver
} // namespace clang

namespace clang::driver::modules {

/// The parsed Standard library module manifest.
struct StdModuleManifest {
  struct Module {
    struct LocalArguments {
      std::vector<std::string> SystemIncludeDirs;
    };

    bool IsStdlib = false;
    std::string LogicalName;
    std::string SourcePath;
    std::optional<LocalArguments> LocalArgs;
  };

  std::vector<Module> Modules;
};

/// Reads the Standard library module manifest at \p ManifestPath.
///
/// Assumes that all file paths specified in the manifest are relative to
/// \p ManifestPath and converts them to absolute.
///
/// \returns The parsed manifest on success; otherwise, a \c llvm::FileError
/// or \c llvm::json::ParseError.
llvm::Expected<StdModuleManifest>
readStdModuleManifest(llvm::StringRef ManifestPath, llvm::vfs::FileSystem &VFS);

/// Constructs compilation inputs for each module listed in the provided
/// Standard library module manifest.
///
/// \param ManifestEntries All entries of the Standard library module manifest.
/// \param C The compilation being built.
/// \param Inputs The input list to which the new module inputs are appended.
void buildStdModuleManifestInputs(
    ArrayRef<StdModuleManifest::Module> ManifestEntries, Compilation &C,
    InputList &Inputs);

/// Scans the compilation inputs for module dependencies and adjusts the
/// compilation to build and supply those modules as required.
///
/// \param C The compilation being built.
/// \param ManifestEntries All entries of the Standard library module manifest.
void runModulesDriver(Compilation &C,
                      ArrayRef<StdModuleManifest::Module> ManifestEntries);

} // namespace clang::driver::modules

#endif // LLVM_CLANG_DRIVER_MODULESDRIVER_H
