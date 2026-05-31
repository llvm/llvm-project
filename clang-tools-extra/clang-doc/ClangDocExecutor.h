//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines a tool executor for clang-doc that runs frontend actions
// on all TUs in the compilation database using the llvm::parallel executor,
// sharing its worker thread pool across phases.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_DOC_CLANGDOCEXECUTOR_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_DOC_CLANGDOCEXECUTOR_H

#include "clang/Tooling/ArgumentsAdjusters.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Execution.h"
#include <optional>

namespace clang::doc {

/// Executes given frontend actions on all files/TUs in the compilation database
/// using the llvm::parallel executor. Tool results are deduplicated by key.
class ClangDocExecutor : public tooling::ToolExecutor {
public:
  static const char *ExecutorName;

  /// Init with a \p CompilationDatabase. Uses \p ThreadCount threads to execute
  /// the actions on all files in parallel; if \p ThreadCount is 0, uses
  /// llvm::hardware_concurrency.
  ClangDocExecutor(const tooling::CompilationDatabase &Compilations,
                   unsigned ThreadCount,
                   std::shared_ptr<PCHContainerOperations> PCHContainerOps =
                       std::make_shared<PCHContainerOperations>());

  /// Init with a \p CommonOptionsParser. The executor takes ownership of
  /// \p Options.
  ClangDocExecutor(tooling::CommonOptionsParser Options, unsigned ThreadCount,
                   std::shared_ptr<PCHContainerOperations> PCHContainerOps =
                       std::make_shared<PCHContainerOperations>());

  StringRef getExecutorName() const override { return ExecutorName; }

  using tooling::ToolExecutor::execute;

  llvm::Error execute(
      llvm::ArrayRef<std::pair<std::unique_ptr<tooling::FrontendActionFactory>,
                               tooling::ArgumentsAdjuster>>
          Actions) override;

  tooling::ExecutionContext *getExecutionContext() override { return &Context; }

  tooling::ToolResults *getToolResults() override { return Results.get(); }

  void mapVirtualFile(StringRef FilePath, StringRef Content) override {
    OverlayFiles[FilePath] = std::string(Content);
  }

private:
  std::optional<tooling::CommonOptionsParser> OptionsParser;
  const tooling::CompilationDatabase &Compilations;
  std::vector<std::string> SourcePaths;
  std::unique_ptr<tooling::ToolResults> Results;
  tooling::ExecutionContext Context;
  llvm::StringMap<std::string> OverlayFiles;
  unsigned ThreadCount;
};

} // namespace clang::doc

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_DOC_CLANGDOCEXECUTOR_H
