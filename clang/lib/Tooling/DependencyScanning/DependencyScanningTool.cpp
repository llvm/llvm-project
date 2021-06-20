//===- DependencyScanningTool.cpp - clang-scan-deps service ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Tooling/DependencyScanning/DependencyScanningTool.h"
#include "clang/Frontend/Utils.h"

namespace clang{
namespace tooling{
namespace dependencies{

std::vector<std::string> FullDependencies::getAdditionalArgs(
    std::function<StringRef(ModuleID)> LookupPCMPath,
    std::function<const ModuleDeps &(ModuleID)> LookupModuleDeps) const {
  std::vector<std::string> Ret = getAdditionalArgsWithoutModulePaths();

  std::vector<std::string> PCMPaths;
  std::vector<std::string> ModMapPaths;
  dependencies::detail::collectPCMAndModuleMapPaths(
      ClangModuleDeps, LookupPCMPath, LookupModuleDeps, PCMPaths, ModMapPaths);
  for (const std::string &PCMPath : PCMPaths)
    Ret.push_back("-fmodule-file=" + PCMPath);
  for (const std::string &ModMapPath : ModMapPaths)
    Ret.push_back("-fmodule-map-file=" + ModMapPath);

  return Ret;
}

std::vector<std::string>
FullDependencies::getAdditionalArgsWithoutModulePaths() const {
  std::vector<std::string> Args{
      "-fno-implicit-modules",
      "-fno-implicit-module-maps",
  };

  for (const PrebuiltModuleDep &PMD : PrebuiltModuleDeps) {
    Args.push_back("-fmodule-file=" + PMD.ModuleName + "=" + PMD.PCMFile);
    Args.push_back("-fmodule-map-file=" + PMD.ModuleMapFile);
  }

  return Args;
}

DependencyScanningTool::DependencyScanningTool(
    DependencyScanningService &Service)
    : Worker(Service) {}

llvm::Expected<std::string> DependencyScanningTool::getDependencyFile(
    const tooling::CompilationDatabase &Compilations, StringRef CWD) {
  /// Prints out all of the gathered dependencies into a string.
  class MakeDependencyPrinterConsumer : public DependencyConsumer {
  public:
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

  // We expect a single command here because if a source file occurs multiple
  // times in the original CDB, then `computeDependencies` would run the
  // `DependencyScanningAction` once for every time the input occured in the
  // CDB. Instead we split up the CDB into single command chunks to avoid this
  // behavior.
  assert(Compilations.getAllCompileCommands().size() == 1 &&
         "Expected a compilation database with a single command!");
  std::string Input = Compilations.getAllCompileCommands().front().Filename;

  MakeDependencyPrinterConsumer Consumer;
  auto Result = Worker.computeDependencies(Input, CWD, Compilations, Consumer);
  if (Result)
    return std::move(Result);
  std::string Output;
  Consumer.printDependencies(Output);
  return Output;
}

llvm::Expected<FullDependenciesResult>
DependencyScanningTool::getFullDependencies(
    const tooling::CompilationDatabase &Compilations, StringRef CWD,
    const llvm::StringSet<> &AlreadySeen) {
  class FullDependencyPrinterConsumer : public DependencyConsumer {
  public:
    FullDependencyPrinterConsumer(const llvm::StringSet<> &AlreadySeen)
        : AlreadySeen(AlreadySeen) {}

    void
    handleDependencyOutputOpts(const DependencyOutputOptions &Opts) override {}

    void handleFileDependency(StringRef File) override {
      Dependencies.push_back(std::string(File));
    }

    void handlePrebuiltModuleDependency(PrebuiltModuleDep PMD) override {
      PrebuiltModuleDeps.emplace_back(std::move(PMD));
    }

    void handleModuleDependency(ModuleDeps MD) override {
      ClangModuleDeps[MD.ID.ContextHash + MD.ID.ModuleName] = std::move(MD);
    }

    void handleContextHash(std::string Hash) override {
      ContextHash = std::move(Hash);
    }

    FullDependenciesResult getFullDependencies() const {
      FullDependencies FD;

      FD.ID.ContextHash = std::move(ContextHash);

      FD.FileDeps.assign(Dependencies.begin(), Dependencies.end());

      for (auto &&M : ClangModuleDeps) {
        auto &MD = M.second;
        if (MD.ImportedByMainFile)
          FD.ClangModuleDeps.push_back(MD.ID);
      }

      FD.PrebuiltModuleDeps = std::move(PrebuiltModuleDeps);

      FullDependenciesResult FDR;

      for (auto &&M : ClangModuleDeps) {
        // TODO: Avoid handleModuleDependency even being called for modules
        //   we've already seen.
        if (AlreadySeen.count(M.first))
          continue;
        FDR.DiscoveredModules.push_back(std::move(M.second));
      }

      FDR.FullDeps = std::move(FD);
      return FDR;
    }

  private:
    std::vector<std::string> Dependencies;
    std::vector<PrebuiltModuleDep> PrebuiltModuleDeps;
    std::unordered_map<std::string, ModuleDeps> ClangModuleDeps;
    std::string ContextHash;
    std::vector<std::string> OutputPaths;
    const llvm::StringSet<> &AlreadySeen;
  };

  // We expect a single command here because if a source file occurs multiple
  // times in the original CDB, then `computeDependencies` would run the
  // `DependencyScanningAction` once for every time the input occured in the
  // CDB. Instead we split up the CDB into single command chunks to avoid this
  // behavior.
  assert(Compilations.getAllCompileCommands().size() == 1 &&
         "Expected a compilation database with a single command!");
  std::string Input = Compilations.getAllCompileCommands().front().Filename;

  FullDependencyPrinterConsumer Consumer(AlreadySeen);
  llvm::Error Result =
      Worker.computeDependencies(Input, CWD, Compilations, Consumer);
  if (Result)
    return std::move(Result);
  return Consumer.getFullDependencies();
}

} // end namespace dependencies
} // end namespace tooling
} // end namespace clang
