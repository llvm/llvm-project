//===--- ScanAndUpdateArgs.h - Util for CC1 Dependency Scanning -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_DRIVER_SCANANDUPDATEARGS_H
#define LLVM_CLANG_DRIVER_SCANANDUPDATEARGS_H

#include "clang/Basic/LLVM.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

namespace llvm {
class StringSaver;
class PrefixMapper;

namespace cas {
class ObjectStore;
class CASID;
} // namespace cas
} // namespace llvm

namespace clang {

class CASOptions;
class CompilerInvocation;
class DiagnosticConsumer;

namespace tooling {
namespace dependencies {
class DependencyScanningTool;

/// Apply CAS inputs for compilation caching to the given invocation, if
/// enabled.
void configureInvocationForCaching(CompilerInvocation &CI, CASOptions CASOpts,
                                   std::string RootID, std::string WorkingDir,
                                   bool ProduceIncludeTree);

struct DepscanPrefixMapping {
  Optional<std::string> NewSDKPath;
  Optional<std::string> NewToolchainPath;
  SmallVector<std::string> PrefixMap;

  /// Add path mappings from the current path in \p Invocation to the new path
  /// from \c DepscanPrefixMapping to the \p Mapper.
  llvm::Error configurePrefixMapper(const CompilerInvocation &Invocation,
                                    llvm::PrefixMapper &Mapper) const;

  /// Apply the mappings from \p Mapper to \p Invocation.
  static void remapInvocationPaths(CompilerInvocation &Invocation,
                                   llvm::PrefixMapper &Mapper);
};
} // namespace dependencies
} // namespace tooling

Expected<llvm::cas::CASID> scanAndUpdateCC1InlineWithTool(
    tooling::dependencies::DependencyScanningTool &Tool,
    DiagnosticConsumer &DiagsConsumer, raw_ostream *VerboseOS,
    CompilerInvocation &Invocation, StringRef WorkingDirectory,
    const tooling::dependencies::DepscanPrefixMapping &PrefixMapping,
    llvm::cas::ObjectStore &DB);

} // end namespace clang

#endif
