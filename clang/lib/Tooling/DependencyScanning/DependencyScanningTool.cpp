//===- DependencyScanningTool.cpp - clang-scan-deps service ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Tooling/DependencyScanning/DependencyScanningTool.h"
#include "clang/Frontend/Utils.h"
#include "llvm/Support/JSON.h"

static llvm::json::Array toJSON(const llvm::StringSet<> &Set) {
  llvm::json::Array Ret;
  for (auto &&I : Set) {
    Ret.push_back(std::string(I.getKey()));
  }
  return Ret;
}

namespace clang{
namespace tooling{
namespace dependencies{

DependencyScanningTool::DependencyScanningTool(
    DependencyScanningService &Service,
    const tooling::CompilationDatabase &Compilations)
    : Format(Service.getFormat()), Worker(Service), Compilations(Compilations) {
}

llvm::Expected<std::string>
DependencyScanningTool::getDependencyFile(const std::string &Input,
                                          StringRef CWD) {
  /// Prints out all of the gathered dependencies into a string.
  class MakeDependencyPrinterConsumer : public DependencyConsumer {
  public:
    void handleFileDependency(const DependencyOutputOptions &Opts,
                              StringRef File) override {
      if (!this->Opts)
        this->Opts = std::make_unique<DependencyOutputOptions>(Opts);
      Dependencies.push_back(File);
    }

    void handleModuleDependency(ModuleDeps MD) override {
      // These are ignored for the make format as it can't support the full
      // set of deps, and handleFileDependency handles enough for implicitly
      // built modules to work.
    }

    void handleContextHash(std::string Hash) override {}

    void printDependencies(std::string &S) {
      if (!Opts)
        return;

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

  class FullDependencyPrinterConsumer : public DependencyConsumer {
  public:
    void handleFileDependency(const DependencyOutputOptions &Opts,
                              StringRef File) override {
      Dependencies.push_back(File);
    }

    void handleModuleDependency(ModuleDeps MD) override {
      ModuleDeps[MD.ContextHash + MD.ModuleName] = std::move(MD);
    }

    void handleContextHash(std::string Hash) override {
      ContextHash = std::move(Hash);
    }

    void printDependencies(std::string &S, StringRef MainFile) {
      // Sort the modules by name to get a deterministic order.
      std::vector<StringRef> Modules;
      for (auto &&Dep : ModuleDeps)
        Modules.push_back(Dep.first);
      std::sort(Modules.begin(), Modules.end());

      llvm::raw_string_ostream OS(S);

      using namespace llvm::json;

      Array Imports;
      for (auto &&ModName : Modules) {
        auto &MD = ModuleDeps[ModName];
        if (MD.ImportedByMainFile)
          Imports.push_back(MD.ModuleName);
      }

      Array Mods;
      for (auto &&ModName : Modules) {
        auto &MD = ModuleDeps[ModName];
        Object Mod{
            {"name", MD.ModuleName},
            {"file-deps", toJSON(MD.FileDeps)},
            {"clang-module-deps", toJSON(MD.ClangModuleDeps)},
            {"clang-modulemap-file", MD.ClangModuleMapFile},
        };
        Mods.push_back(std::move(Mod));
      }

      Object O{
          {"input-file", MainFile},
          {"clang-context-hash", ContextHash},
          {"file-deps", Dependencies},
          {"clang-module-deps", std::move(Imports)},
          {"clang-modules", std::move(Mods)},
      };

      S = llvm::formatv("{0:2},\n", Value(std::move(O))).str();
      return;
    }

  private:
    std::vector<std::string> Dependencies;
    std::unordered_map<std::string, ModuleDeps> ModuleDeps;
    std::string ContextHash;
  };

  if (Format == ScanningOutputFormat::Make) {
    MakeDependencyPrinterConsumer Consumer;
    auto Result =
        Worker.computeDependencies(Input, CWD, Compilations, Consumer);
    if (Result)
      return std::move(Result);
    std::string Output;
    Consumer.printDependencies(Output);
    return Output;
  } else {
    FullDependencyPrinterConsumer Consumer;
    auto Result =
        Worker.computeDependencies(Input, CWD, Compilations, Consumer);
    if (Result)
      return std::move(Result);
    std::string Output;
    Consumer.printDependencies(Output, Input);
    return Output;
  }
}

} // end namespace dependencies
} // end namespace tooling
} // end namespace clang
