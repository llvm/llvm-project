//===- CachedDiagnostics.h - Serializing diagnostics for caching-*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_DRIVER_CACHEDDIAGNOSTICS_H
#define LLVM_CLANG_TOOLS_DRIVER_CACHEDDIAGNOSTICS_H

#include "clang/Basic/LLVM.h"
#include "llvm/Support/PrefixMapper.h"

namespace clang {
class DiagnosticConsumer;
class DiagnosticsEngine;
class FileManager;

namespace cas {

class CachingDiagnosticsProcessor {
public:
  /// The \p Mapper is used for de-canonicalizing the paths of diagnostics
  /// before rendering them.
  CachingDiagnosticsProcessor(llvm::PrefixMapper Mapper, FileManager &FileMgr);
  ~CachingDiagnosticsProcessor();

  /// Insert a diagnostic consumer for capturing diagnostics before starting a
  /// normal compilation.
  void insertDiagConsumer(DiagnosticsEngine &Diags);
  /// Remove the diagnostic consumer after the normal compilation finished.
  void removeDiagConsumer(DiagnosticsEngine &Diags);

  /// \returns a serialized buffer of the currently recorded diagnostics, or
  /// \p std::nullopt if there's no diagnostic. The buffer can be passed to
  /// \p replayCachedDiagnostics for rendering the same diagnostics.
  ///
  /// There is no stability guarantee for the format of the buffer, the
  /// expectation is that the buffer will be deserialized only by the same
  /// compiler version that produced it. The format can change without
  /// restrictions.
  ///
  /// The intended use is as implementation detail of compilation caching, where
  /// the diagnostic output is associated with a compilation cache key. A
  /// different compiler version will create different cache keys, which ensures
  /// that the diagnostics buffer will only be read by the same compiler that
  /// produced it.
  Expected<std::optional<std::string>> serializeEmittedDiagnostics();

  /// Used to replay the previously cached diagnostics, after a cache hit.
  llvm::Error replayCachedDiagnostics(StringRef Buffer,
                                      DiagnosticConsumer &Consumer);

private:
  llvm::PrefixMapper Mapper;
  FileManager &FileMgr;

  struct DiagnosticsConsumer;
  std::unique_ptr<DiagnosticsConsumer> DiagConsumer;
};

} // namespace cas
} // namespace clang

#endif // LLVM_CLANG_TOOLS_DRIVER_CACHEDDIAGNOSTICS_H
