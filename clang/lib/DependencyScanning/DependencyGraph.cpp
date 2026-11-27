//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/DependencyScanning/DependencyGraph.h"

#include "clang/Serialization/ASTReader.h"
#include "llvm/ADT/SmallString.h"

using namespace clang;
using namespace clang::dependencies;

void ModuleDeps::forEachFileDep(llvm::function_ref<void(StringRef)> Cb) const {
  SmallString<0> PathBuf;
  PathBuf.reserve(256);
  for (StringRef FileDep : FileDeps) {
    auto ResolvedFileDep =
        ASTReader::ResolveImportedPath(PathBuf, FileDep, FileDepsBaseDir);
    Cb(*ResolvedFileDep);
  }
}

const std::vector<std::string> &ModuleDeps::getBuildArguments() const {
  // FIXME: this operation is not thread safe and is expected to be called
  // on a single thread. Otherwise, it should be protected with a lock.
  assert(!std::holds_alternative<std::monostate>(BuildInfo) &&
         "Using uninitialized ModuleDeps");
  if (const auto *CI = std::get_if<CowCompilerInvocation>(&BuildInfo))
    BuildInfo = CI->getCC1CommandLine();
  return std::get<std::vector<std::string>>(BuildInfo);
}
