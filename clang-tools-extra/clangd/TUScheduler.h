//===--- TUScheduler.h -------------------------------------------*-C++-*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANGD_TUSCHEDULER_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANGD_TUSCHEDULER_H

#include "ClangdUnit.h"
#include "Function.h"
#include "Threading.h"
#include "llvm/ADT/StringMap.h"

namespace clang {
namespace clangd {
/// Returns a number of a default async threads to use for TUScheduler.
/// Returned value is always >= 1 (i.e. will not cause requests to be processed
/// synchronously).
unsigned getDefaultAsyncThreadsCount();

struct InputsAndAST {
  const ParseInputs &Inputs;
  ParsedAST &AST;
};

struct InputsAndPreamble {
  const ParseInputs &Inputs;
  const PreambleData *Preamble;
};

/// Handles running tasks for ClangdServer and managing the resources (e.g.,
/// preambles and ASTs) for opened files.
/// TUScheduler is not thread-safe, only one thread should be providing updates
/// and scheduling tasks.
/// Callbacks are run on a threadpool and it's appropriate to do slow work in
/// them.
class TUScheduler {
public:
  TUScheduler(unsigned AsyncThreadsCount, bool StorePreamblesInMemory,
              ASTParsedCallback ASTCallback);
  ~TUScheduler();

  /// Returns estimated memory usage for each of the currently open files.
  /// The order of results is unspecified.
  std::vector<std::pair<Path, std::size_t>> getUsedBytesPerFile() const;

  /// Schedule an update for \p File. Adds \p File to a list of tracked files if
  /// \p File was not part of it before.
  /// FIXME(ibiryukov): remove the callback from this function.
  void update(PathRef File, ParseInputs Inputs,
              UniqueFunction<void(llvm::Optional<std::vector<DiagWithFixIts>>)>
                  OnUpdated);

  /// Remove \p File from the list of tracked files and schedule removal of its
  /// resources.
  void remove(PathRef File);

  /// Schedule an async read of the AST. \p Action will be called when AST is
  /// ready. The AST passed to \p Action refers to the version of \p File
  /// tracked at the time of the call, even if new updates are received before
  /// \p Action is executed.
  /// If an error occurs during processing, it is forwarded to the \p Action
  /// callback.
  void runWithAST(PathRef File,
                  UniqueFunction<void(llvm::Expected<InputsAndAST>)> Action);

  /// Schedule an async read of the Preamble. Preamble passed to \p Action may
  /// be built for any version of the file, callers must not rely on it being
  /// consistent with the current version of the file.
  /// If an error occurs during processing, it is forwarded to the \p Action
  /// callback.
  void runWithPreamble(
      PathRef File,
      UniqueFunction<void(llvm::Expected<InputsAndPreamble>)> Action);

private:
  /// This class stores per-file data in the Files map.
  struct FileData;

  const bool StorePreamblesInMemory;
  const std::shared_ptr<PCHContainerOperations> PCHOps;
  const ASTParsedCallback ASTCallback;
  Semaphore Barrier;
  llvm::StringMap<std::unique_ptr<FileData>> Files;
  // None when running tasks synchronously and non-None when running tasks
  // asynchronously.
  llvm::Optional<AsyncTaskRunner> Tasks;
};
} // namespace clangd
} // namespace clang

#endif
