//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_DEPENDENCYSCANNING_DEPENDENCYCONSUMER_H
#define LLVM_CLANG_DEPENDENCYSCANNING_DEPENDENCYCONSUMER_H

#include "clang/Basic/LLVM.h"
#include "clang/DependencyScanning/DependencyGraph.h"

#include <optional>
#include <string>
#include <vector>

namespace clang::dependencies {
class DependencyConsumer {
public:
  virtual ~DependencyConsumer() {}

  virtual void handleProvidedAndRequiredStdCXXModules(
      std::optional<P1689ModuleInfo> Provided,
      std::vector<P1689ModuleInfo> Requires) {}

  virtual void handleBuildCommand(Command Cmd) {}

  virtual void
  handleDependencyOutputOpts(const DependencyOutputOptions &Opts) = 0;

  virtual void handleFileDependency(StringRef Filename) = 0;

  virtual void handlePrebuiltModuleDependency(PrebuiltModuleDep PMD) = 0;

  virtual void handleModuleDependency(ModuleDeps MD) = 0;

  virtual void handleDirectModuleDependency(ModuleID MD) = 0;

  virtual void handleVisibleModule(std::string ModuleName) = 0;

  virtual void handleContextHash(std::string Hash) = 0;
};
} // namespace clang::dependencies

#endif // LLVM_CLANG_DEPENDENCYSCANNING_DEPENDENCYCONSUMER_H
