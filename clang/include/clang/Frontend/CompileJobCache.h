//===- CompileJobCache.h ----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_FRONTEND_COMPILEJOBCACHE_H
#define LLVM_CLANG_FRONTEND_COMPILEJOBCACHE_H

#include "clang/Frontend/CompileJobCacheResult.h"

namespace clang {

class CompilerInstance;
class CompilerInvocation;
class DiagnosticsEngine;

// Manage caching and replay of compile jobs.
//
// The high-level model is:
//
//  1. Extract options from the CompilerInvocation:
//       - that can be simulated and
//       - that don't affect the compile job's result.
//  2. Canonicalize the options extracted in (1).
//  3. Compute the result of the compile job using the canonicalized
//     CompilerInvocation, with hooks installed to redirect outputs and
//     enable live-streaming of a running compile job to stdout or stderr.
//       - Compute a cache key.
//       - Check the cache, and run the compile job if there's a cache miss.
//       - Store the result of the compile job in the cache.
//  4. Replay the compile job, using the options extracted in (1).
//
// An example (albeit not yet implemented) is handling options controlling
// output of diagnostics. The CompilerInvocation can be canonicalized to
// serialize the diagnostics to a virtual path (<output>.diag or something).
//
//   - On a cache miss, the compile job runs, and the diagnostics are
//     serialized and stored in the cache per the canonicalized options
//     from (2).
//   - Either way, the diagnostics are replayed according to the options
//     extracted from (1) during (4).
//
// The above will produce the correct output for diagnostics, but the experience
// will be degraded in the common command-line case (emitting to stderr)
// because the diagnostics will not be streamed live. This can be improved:
//
//   - Change (3) to accept a hook: a DiagnosticsConsumer that diagnostics
//     are mirrored to (in addition to canonicalized options from (2)).
//   - If diagnostics would be live-streamed, send in a diagnostics consumer
//     that matches (1). Otherwise, send in an IgnoringDiagnosticsConsumer.
//   - In step (4), only skip replaying the diagnostics if they were already
//     handled.
class CompileJobCache {
public:
  CompileJobCache();
  ~CompileJobCache();

  using OutputKind = clang::cas::CompileJobCacheResult::OutputKind;

  StringRef getPathForOutputKind(OutputKind Kind);

  /// Canonicalize \p Clang.
  ///
  /// \returns status if should exit immediately, otherwise None.
  ///
  /// TODO: Refactor \a cc1_main() so that instead this canonicalizes the
  /// CompilerInvocation before Clang gets access to command-line arguments, to
  /// control what might leak.
  std::optional<int> initialize(CompilerInstance &Clang);

  /// Try looking up a cached result and replaying it.
  ///
  /// \returns status if should exit immediately, otherwise None.
  std::optional<int> tryReplayCachedResult(CompilerInstance &Clang);

  /// Finish writing outputs from a computed result, after a cache miss.
  ///
  /// \returns true if finished successfully.
  bool finishComputedResult(CompilerInstance &Clang, bool Success);

  static llvm::Expected<std::optional<int>>
  replayCachedResult(std::shared_ptr<CompilerInvocation> Invok,
                     StringRef WorkingDir, const llvm::cas::CASID &CacheKey,
                     cas::CompileJobCacheResult &CachedResult,
                     SmallVectorImpl<char> &DiagText,
                     bool WriteOutputAsCASID = false,
                     std::optional<llvm::cas::CASID> *MCOutputID = nullptr);

  class CachingOutputs;

private:
  int reportCachingBackendError(DiagnosticsEngine &Diag, llvm::Error &&E);

  bool CacheCompileJob = false;
  bool DisableCachedCompileJobReplay = false;
  std::optional<llvm::cas::CASID> MCOutputID;

  std::shared_ptr<llvm::cas::ObjectStore> CAS;
  std::shared_ptr<llvm::cas::ActionCache> Cache;
  std::optional<llvm::cas::CASID> ResultCacheKey;

  std::unique_ptr<CachingOutputs> CacheBackend;
};

} // namespace clang

#endif // LLVM_CLANG_FRONTEND_COMPILEJOBCACHE_H
