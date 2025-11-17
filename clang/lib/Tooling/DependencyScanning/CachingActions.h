//===- CachingActions.h -----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLING_DEPENDENCYSCANNING_CACHINGACTIONS_H
#define LLVM_CLANG_TOOLING_DEPENDENCYSCANNING_CACHINGACTIONS_H

#include "clang/Tooling/DependencyScanning/DependencyScanningTool.h"
#include "clang/Tooling/DependencyScanning/ScanAndUpdateArgs.h"

namespace clang::tooling::dependencies {

std::unique_ptr<DependencyActionController>
createIncludeTreeActionController(LookupModuleOutputCallback LookupModuleOutput,
                                  cas::ObjectStore &DB);

/// The PCH recorded file paths with canonical paths, create a VFS that
/// allows remapping back to the non-canonical source paths so that they are
/// found during dep-scanning.
void addReversePrefixMappingFileSystem(const llvm::PrefixMapper &PrefixMapper,
                                       CompilerInstance &ScanInstance);

} // namespace clang::tooling::dependencies
#endif // LLVM_CLANG_TOOLING_DEPENDENCYSCANNING_CACHINGACTIONS_H
