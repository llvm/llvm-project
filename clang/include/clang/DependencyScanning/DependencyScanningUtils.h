//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_DEPENDENCYSCANNING_DEPENDENCYSCANNINGUTILS_H
#define LLVM_CLANG_DEPENDENCYSCANNING_DEPENDENCYSCANNINGUTILS_H

#include "clang/DependencyScanning/DependencyScannerImpl.h"
#include "clang/DependencyScanning/DependencyScanningWorker.h"
#include "clang/DependencyScanning/ModuleDepCollector.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/MapVector.h"
#include <string>
#include <vector>

namespace clang {
namespace dependencies {

/// Graph of modular dependencies.
using ModuleDepsGraph = std::vector<ModuleDeps>;

/// The full dependencies and module graph for a specific input.
struct TranslationUnitDeps {
  /// The graph of direct and transitive modular dependencies.
  ModuleDepsGraph ModuleGraph;

  /// The identifier of the C++20 module this translation unit exports.
  ///
  /// If the translation unit is not a module then \c ID.ModuleName is empty.
  ModuleID ID;

  /// A collection of absolute paths to files that this translation unit
  /// directly depends on, not including transitive dependencies.
  std::vector<std::string> FileDeps;

  /// A collection of prebuilt modules this translation unit directly depends
  /// on, not including transitive dependencies.
  std::vector<PrebuiltModuleDep> PrebuiltModuleDeps;

  /// A list of modules this translation unit directly depends on, not including
  /// transitive dependencies.
  ///
  /// This may include modules with a different context hash when it can be
  /// determined that the differences are benign for this compilation.
  std::vector<ModuleID> ClangModuleDeps;

  /// A list of module names that are visible to this translation unit. This
  /// includes both direct and transitive module dependencies.
  std::vector<std::string> VisibleModules;

  /// A list of the C++20 named modules this translation unit depends on.
  std::vector<std::string> NamedModuleDeps;

  /// The sequence of commands required to build the translation unit. Commands
  /// should be executed in order.
  ///
  /// FIXME: If we add support for multi-arch builds in clang-scan-deps, we
  /// should make the dependencies between commands explicit to enable parallel
  /// builds of each architecture.
  std::vector<Command> Commands;

  /// Deprecated driver command-line. This will be removed in a future version.
  std::vector<std::string> DriverCommandLine;
};

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

  std::string lookupModuleOutput(const ModuleDeps &MD,
                                 ModuleOutputKind Kind) override {
    return LookupModuleOutput(MD, Kind);
  }

private:
  LookupModuleOutputCallback LookupModuleOutput;
};

} // end namespace dependencies
} // end namespace clang

#endif // LLVM_CLANG_DEPENDENCYSCANNING_DEPENDENCYSCANNINGUTILS_H
