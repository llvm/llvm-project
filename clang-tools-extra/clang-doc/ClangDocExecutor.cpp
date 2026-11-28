//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ClangDocExecutor.h"
#include "clang/Basic/DiagnosticOptions.h"
#include "clang/Frontend/TextDiagnosticPrinter.h"
#include "clang/Tooling/AllTUsExecution.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/Support/Parallel.h"
#include "llvm/Support/Regex.h"
#include "llvm/Support/Threading.h"
#include "llvm/Support/VirtualFileSystem.h"
#include "llvm/Support/raw_ostream.h"

#include <atomic>

namespace clang::doc {

using namespace clang::tooling;

const char *ClangDocExecutor::ExecutorName = "ClangDocExecutor";

namespace {
ArgumentsAdjuster getDefaultArgumentsAdjusters() {
  return combineAdjusters(
      getClangStripOutputAdjuster(),
      combineAdjusters(getClangSyntaxOnlyAdjuster(),
                       getClangStripDependencyFileAdjuster()));
}

class ThreadSafeToolResults : public ToolResults {
public:
  void addResult(StringRef Key, StringRef Value) override {
    std::unique_lock<std::mutex> LockGuard(Mutex);
    Results.addResult(Key, Value);
  }

  std::vector<std::pair<llvm::StringRef, llvm::StringRef>>
  AllKVResults() override {
    return Results.AllKVResults();
  }

  void forEachResult(llvm::function_ref<void(StringRef Key, StringRef Value)>
                         Callback) override {
    Results.forEachResult(Callback);
  }

private:
  InMemoryToolResults Results;
  std::mutex Mutex;
};

} // namespace

ClangDocExecutor::ClangDocExecutor(
    const CompilationDatabase &Compilations, unsigned ThreadCount,
    std::shared_ptr<PCHContainerOperations> PCHContainerOps)
    : Compilations(Compilations), Results(new ThreadSafeToolResults),
      Context(Results.get()), ThreadCount(ThreadCount) {}

ClangDocExecutor::ClangDocExecutor(
    CommonOptionsParser Options, unsigned ThreadCount,
    std::shared_ptr<PCHContainerOperations> PCHContainerOps)
    : OptionsParser(std::move(Options)),
      Compilations(OptionsParser->getCompilations()),
      SourcePaths(OptionsParser->getSourcePathList()),
      Results(new ThreadSafeToolResults), Context(Results.get()),
      ThreadCount(ThreadCount) {}

llvm::Error ClangDocExecutor::execute(
    llvm::ArrayRef<
        std::pair<std::unique_ptr<FrontendActionFactory>, ArgumentsAdjuster>>
        Actions) {
  if (Actions.empty())
    return llvm::createStringError("No action to execute.");

  // Just copy the behavior of the ToolExecutor, and only support 1 action.
  if (Actions.size() != 1)
    return llvm::createStringError(
        "Only support executing exactly 1 action at this point.");

  std::string ErrorMsg;
  std::mutex TUMutex;
  auto AppendError = [&](llvm::Twine Err) {
    std::unique_lock<std::mutex> LockGuard(TUMutex);
    ErrorMsg += Err.str();
  };

  auto Log = [&](llvm::Twine Msg) {
    std::unique_lock<std::mutex> LockGuard(TUMutex);
    llvm::errs() << Msg << "\n";
  };

  // We follow the existing libTooling conventions for executors.
  // The all-TUs executor processes every file in the compilation database.
  // The standalone executor processes the files named on the command line.
  std::vector<std::string> Inputs = tooling::ExecutorName == "all-TUs"
                                        ? Compilations.getAllFiles()
                                        : SourcePaths;
  std::vector<std::string> Files;
  llvm::Regex RegexFilter(Filter);
  for (const auto &File : Inputs) {
    if (RegexFilter.match(File))
      Files.push_back(File);
  }
  // Add a counter to track the progress.
  const std::string TotalNumStr = std::to_string(Files.size());
  std::atomic<uint32_t> Counter{0};

  auto &Action = Actions.front();

  // Run the per-TU actions on the llvm::parallel executor.
  llvm::parallel::strategy = llvm::hardware_concurrency(ThreadCount);
  llvm::parallelFor(0, Files.size(), [&](size_t Index) {
    std::string Path = Files[Index];
    Log("[" + std::to_string(++Counter) + "/" + TotalNumStr +
        "] Processing file " + Path);

    // Give each thread its own  VFS to allow different concurrent working
    // directories.
    IntrusiveRefCntPtr<llvm::vfs::FileSystem> FS =
        llvm::vfs::createPhysicalFileSystem();
    ClangTool Tool(Compilations, {Path},
                   std::make_shared<PCHContainerOperations>(), FS);
    Tool.appendArgumentsAdjuster(Action.second);
    Tool.appendArgumentsAdjuster(getDefaultArgumentsAdjusters());
    for (const auto &FileAndContent : OverlayFiles)
      Tool.mapVirtualFile(FileAndContent.first(), FileAndContent.second);
    std::string DiagBuffer;
    llvm::raw_string_ostream DiagOS(DiagBuffer);
    DiagnosticOptions DiagOpts;
    TextDiagnosticPrinter DiagPrinter(DiagOS, DiagOpts);
    Tool.setDiagnosticConsumer(&DiagPrinter);
    if (Tool.run(Action.first.get()))
      AppendError(llvm::Twine("Failed to run action on ") + Path + "\n");
    if (!DiagBuffer.empty())
      Log(DiagBuffer);
  });

  if (!ErrorMsg.empty())
    return llvm::createStringError(ErrorMsg);

  return llvm::Error::success();
}

} // namespace clang::doc
