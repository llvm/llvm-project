//===- InMemoryModuleCache.h - In-memory cache for modules ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_SERIALIZATION_INMEMORYMODULECACHE_H
#define LLVM_CLANG_SERIALIZATION_INMEMORYMODULECACHE_H

#include "llvm/ADT/IntrusiveRefCntPtr.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/MemoryBuffer.h"
#include <memory>

namespace clang {

/// In-memory cache for modules.
///
/// This is a cache for modules for use across a compilation, sharing state
/// between the CompilerInstances in a modules build. It must be shared by each
/// CompilerInstance, ASTReader, ASTWriter, and ModuleManager that are
/// coordinating.
///
/// Critically, it ensures that a single process has a consistent view of each
/// implicitly-built PCM. This is used by \a CompilerInstance when building PCMs
/// to ensure that each \a ModuleManager sees the same files.
class InMemoryModuleCache : public llvm::RefCountedBase<InMemoryModuleCache> {
  struct PCM {
    /// The contents of the PCM as produced by \c ASTWriter.
    std::unique_ptr<llvm::MemoryBuffer> Buffer;

    /// The size of this PCM. This may be different from the size of \c Buffer
    /// when it's wrapped in an object file.
    off_t Size = 0;

    /// The modification time of this PCM.
    time_t ModTime = 0;

    /// Track whether this PCM is known to be good (either built or
    /// successfully imported by a CompilerInstance/ASTReader using this
    /// cache).
    bool IsFinal = false;

    PCM() = default;
    PCM(std::unique_ptr<llvm::MemoryBuffer> Buffer, off_t Size, time_t ModTime)
        : Buffer(std::move(Buffer)), Size(Size), ModTime(ModTime) {}
  };

  /// Cache of buffers.
  llvm::StringMap<PCM> PCMs;

public:
  /// There are four states for a PCM.  It must monotonically increase.
  ///
  ///  1. Unknown: the PCM has neither been read from disk nor built.
  ///  2. Tentative: the PCM has been read from disk but not yet imported or
  ///     built.  It might work.
  ///  3. ToBuild: the PCM read from disk did not work but a new one has not
  ///     been built yet.
  ///  4. Final: indicating that the current PCM was either built in this
  ///     process or has been successfully imported.
  enum State { Unknown, Tentative, ToBuild, Final };

  /// Get the state of the PCM.
  State getPCMState(llvm::StringRef Filename) const;

  /// Store the PCM under the Filename.
  ///
  /// \pre state is Unknown
  /// \post state is Tentative
  /// \return a reference to the buffer as a convenience.
  llvm::MemoryBuffer &addPCM(llvm::StringRef Filename,
                             std::unique_ptr<llvm::MemoryBuffer> Buffer,
                             off_t Size, time_t ModTime);

  /// Store a just-built PCM under the Filename.
  ///
  /// \pre state is Unknown or ToBuild.
  /// \pre state is not Tentative.
  /// \return a reference to the buffer as a convenience.
  llvm::MemoryBuffer &addBuiltPCM(llvm::StringRef Filename,
                                  std::unique_ptr<llvm::MemoryBuffer> Buffer,
                                  off_t Size, time_t ModTime);

  /// Try to remove a buffer from the cache.  No effect if state is Final.
  ///
  /// \pre state is Tentative/Final.
  /// \post Tentative => ToBuild or Final => Final.
  /// \return false on success, i.e. if Tentative => ToBuild.
  bool tryToDropPCM(llvm::StringRef Filename);

  /// Mark a PCM as final.
  ///
  /// \pre state is Tentative or Final.
  /// \post state is Final.
  void finalizePCM(llvm::StringRef Filename);

  /// Get a pointer to the PCM if it exists and set \c Size and \c ModTime to
  /// its on-disk size and modification time. Otherwise, return nullptr and
  /// don't change \c Size and \c ModTime.
  llvm::MemoryBuffer *lookupPCM(llvm::StringRef Filename, off_t &Size,
                                time_t &ModTime) const;

  /// Check whether the PCM is final and has been shown to work.
  ///
  /// \return true iff state is Final.
  bool isPCMFinal(llvm::StringRef Filename) const;

  /// Check whether the PCM is waiting to be built.
  ///
  /// \return true iff state is ToBuild.
  bool shouldBuildPCM(llvm::StringRef Filename) const;
};

} // end namespace clang

#endif // LLVM_CLANG_SERIALIZATION_INMEMORYMODULECACHE_H
