//===- DependencyScanningTool.cpp - clang-scan-deps service ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Tooling/DependencyScanning/DependencyScanningTool.h"
#include "clang/Frontend/Utils.h"
#include <optional>

using namespace clang;
using namespace tooling;
using namespace dependencies;

DependencyScanningTool::DependencyScanningTool(
    DependencyScanningService &Service,
    llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> FS)
    : Worker(Service, std::move(FS)) {}

llvm::Expected<std::string> DependencyScanningTool::getDependencyFile(
    const std::vector<std::string> &CommandLine, StringRef CWD) {
  /// Prints out all of the gathered dependencies into a string.
  class MakeDependencyPrinterConsumer : public DependencyConsumer {
  public:
    void handleBuildCommand(Command) override {}

    void
    handleDependencyOutputOpts(const DependencyOutputOptions &Opts) override {
      this->Opts = std::make_unique<DependencyOutputOptions>(Opts);
    }

    void handleFileDependency(StringRef File) override {
      Dependencies.push_back(std::string(File));
    }

    void handlePrebuiltModuleDependency(PrebuiltModuleDep PMD) override {
      // Same as `handleModuleDependency`.
    }

    void handleModuleDependency(ModuleDeps MD) override {
      // These are ignored for the make format as it can't support the full
      // set of deps, and handleFileDependency handles enough for implicitly
      // built modules to work.
    }

    void handleContextHash(std::string Hash) override {}

    std::string lookupModuleOutput(const ModuleID &ID,
                                   ModuleOutputKind Kind) override {
      llvm::report_fatal_error("unexpected call to lookupModuleOutput");
    }

    void printDependencies(std::string &S) {
      assert(Opts && "Handled dependency output options.");

      class DependencyPrinter : public DependencyFileGenerator {
      public:
        DependencyPrinter(DependencyOutputOptions &Opts,
                          ArrayRef<std::string> Dependencies)
            : DependencyFileGenerator(Opts) {
          for (const auto &Dep : Dependencies)
            addDependency(Dep);
        }

        void printDependencies(std::string &S) {
          llvm::raw_string_ostream OS(S);
          outputDependencyFile(OS);
        }
      };

      DependencyPrinter Generator(*Opts, Dependencies);
      Generator.printDependencies(S);
    }

  private:
    std::unique_ptr<DependencyOutputOptions> Opts;
    std::vector<std::string> Dependencies;
  };

  MakeDependencyPrinterConsumer Consumer;
  auto Result = Worker.computeDependencies(CWD, CommandLine, Consumer);
  if (Result)
    return std::move(Result);
  std::string Output;
  Consumer.printDependencies(Output);
  return Output;
}

llvm::Expected<TranslationUnitDeps>
DependencyScanningTool::getTranslationUnitDependencies(
    const std::vector<std::string> &CommandLine, StringRef CWD,
    const llvm::StringSet<> &AlreadySeen,
    LookupModuleOutputCallback LookupModuleOutput) {
  FullDependencyConsumer Consumer(AlreadySeen, LookupModuleOutput);
  llvm::Error Result = Worker.computeDependencies(CWD, CommandLine, Consumer);
  if (Result)
    return std::move(Result);
  return Consumer.takeTranslationUnitDeps();
}

llvm::Expected<ModuleDepsGraph> DependencyScanningTool::getModuleDependencies(
    StringRef ModuleName, const std::vector<std::string> &CommandLine,
    StringRef CWD, const llvm::StringSet<> &AlreadySeen,
    LookupModuleOutputCallback LookupModuleOutput) {
  FullDependencyConsumer Consumer(AlreadySeen, LookupModuleOutput);
  llvm::Error Result =
      Worker.computeDependencies(CWD, CommandLine, Consumer, ModuleName);
  if (Result)
    return std::move(Result);
  return Consumer.takeModuleGraphDeps();
}

TranslationUnitDeps FullDependencyConsumer::takeTranslationUnitDeps() {
  TranslationUnitDeps TU;

  TU.ID.ContextHash = std::move(ContextHash);
  TU.FileDeps = std::move(Dependencies);
  TU.PrebuiltModuleDeps = std::move(PrebuiltModuleDeps);
  TU.Commands = std::move(Commands);

  for (auto &&M : ClangModuleDeps) {
    auto &MD = M.second;
    if (MD.ImportedByMainFile)
      TU.ClangModuleDeps.push_back(MD.ID);
    // TODO: Avoid handleModuleDependency even being called for modules
    //   we've already seen.
    if (AlreadySeen.count(M.first))
      continue;
    TU.ModuleGraph.push_back(std::move(MD));
  }

  return TU;
}

ModuleDepsGraph FullDependencyConsumer::takeModuleGraphDeps() {
  ModuleDepsGraph ModuleGraph;

  for (auto &&M : ClangModuleDeps) {
    auto &MD = M.second;
    // TODO: Avoid handleModuleDependency even being called for modules
    //   we've already seen.
    if (AlreadySeen.count(M.first))
      continue;
    ModuleGraph.push_back(std::move(MD));
  }

  return ModuleGraph;
}
