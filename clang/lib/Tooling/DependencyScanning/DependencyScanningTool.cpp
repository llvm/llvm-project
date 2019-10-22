//===- DependencyScanningTool.cpp - clang-scan-deps service ------------===//
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

DependencyScanningTool::DependencyScanningTool(DependencyScanningService &Service,
const tooling::CompilationDatabase &Compilations) : Worker(Service), Compilations(Compilations) {}

llvm::Expected<std::string> DependencyScanningTool::getDependencyFile(const std::string &Input,
                                              StringRef CWD) {
  /// Prints out all of the gathered dependencies into a string.
  class DependencyPrinterConsumer : public DependencyConsumer {
  public:
    void handleFileDependency(const DependencyOutputOptions &Opts,
                              StringRef File) override {
      if (!this->Opts)
        this->Opts = std::make_unique<DependencyOutputOptions>(Opts);
      Dependencies.push_back(File);
    }

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

  DependencyPrinterConsumer Consumer;
  auto Result =
      Worker.computeDependencies(Input, CWD, Compilations, Consumer);
  if (Result)
    return std::move(Result);
  std::string Output;
  Consumer.printDependencies(Output);
  return Output;
}

} // end namespace dependencies
} // end namespace tooling
} // end namespace clang
