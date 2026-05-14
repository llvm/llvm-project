//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_DEPENDENCYSCANNING_DEPENDENCYSCANNINGUTILS_H
#define LLVM_CLANG_DEPENDENCYSCANNING_DEPENDENCYSCANNINGUTILS_H

#include "clang/DependencyScanning/DependencyActionController.h"
#include "clang/DependencyScanning/DependencyScannerImpl.h"
#include "clang/DependencyScanning/DependencyScanningWorker.h"
#include "clang/DependencyScanning/ModuleDepCollector.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/MapVector.h"
#include <string>
#include <vector>

namespace clang {
namespace dependencies {
class FullDependencyConsumer : public DependencyConsumer {
public:
  FullDependencyConsumer(const llvm::DenseSet<ModuleID> &AlreadySeen)
      : AlreadySeen(AlreadySeen) {}

  void handleBuildCommand(Command Cmd) override {
    Commands.push_back(std::move(Cmd));
  }

  void handleDependencyOutputOpts(const DependencyOutputOptions &) override {}

  void handleFileDependency(StringRef File) override {
    Dependencies.push_back(std::string(File));
  }

  void handlePrebuiltModuleDependency(PrebuiltModuleDep PMD) override {
    PrebuiltModuleDeps.emplace_back(std::move(PMD));
  }

  void handleModuleDependency(ModuleDeps MD) override {
    ClangModuleDeps[MD.ID] = std::move(MD);
  }

  void handleDirectModuleDependency(ModuleID ID) override {
    DirectModuleDeps.push_back(ID);
  }

  void handleVisibleModule(std::string ModuleName) override {
    VisibleModules.push_back(ModuleName);
  }

  void handleContextHash(std::string Hash) override {
    ContextHash = std::move(Hash);
  }

  void handleProvidedAndRequiredStdCXXModules(
      std::optional<P1689ModuleInfo> Provided,
      std::vector<P1689ModuleInfo> Requires) override {
    ModuleName = Provided ? Provided->ModuleName : "";
    llvm::transform(Requires, std::back_inserter(NamedModuleDeps),
                    [](const auto &Module) { return Module.ModuleName; });
  }

  TranslationUnitDeps takeTranslationUnitDeps();

private:
  std::vector<std::string> Dependencies;
  std::vector<PrebuiltModuleDep> PrebuiltModuleDeps;
  llvm::MapVector<ModuleID, ModuleDeps> ClangModuleDeps;
  std::string ModuleName;
  std::vector<std::string> NamedModuleDeps;
  std::vector<ModuleID> DirectModuleDeps;
  std::vector<std::string> VisibleModules;
  std::vector<Command> Commands;
  std::string ContextHash;
  const llvm::DenseSet<ModuleID> &AlreadySeen;
};

/// A callback to lookup module outputs for "-fmodule-file=", "-o" etc.
using LookupModuleOutputCallback =
    llvm::function_ref<std::string(const ModuleDeps &, ModuleOutputKind)>;

/// A simple dependency action controller that uses a callback. If no callback
/// is provided, it is assumed that looking up module outputs is unreachable.
class CallbackActionController : public DependencyActionController {
public:
  virtual ~CallbackActionController();

  static std::string lookupUnreachableModuleOutput(const ModuleDeps &MD,
                                                   ModuleOutputKind Kind) {
    llvm::report_fatal_error("unexpected call to lookupModuleOutput");
  };

  CallbackActionController(LookupModuleOutputCallback LMO)
      : LookupModuleOutput(std::move(LMO)) {
    if (!LookupModuleOutput) {
      LookupModuleOutput = lookupUnreachableModuleOutput;
    }
  }

  std::unique_ptr<DependencyActionController> clone() const override {
    return std::make_unique<CallbackActionController>(LookupModuleOutput);
  }

  std::string lookupModuleOutput(const ModuleDeps &MD,
                                 ModuleOutputKind Kind) override {
    return LookupModuleOutput(MD, Kind);
  }

protected:
  LookupModuleOutputCallback LookupModuleOutput;
};

} // end namespace dependencies
} // end namespace clang

#endif // LLVM_CLANG_DEPENDENCYSCANNING_DEPENDENCYSCANNINGUTILS_H
