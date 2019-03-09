//===- InMemoryModuleCache.h - In-memory cache for modules ------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_SERIALIZATION_INMEMORYMODULECACHE_H
#define LLVM_CLANG_SERIALIZATION_INMEMORYMODULECACHE_H

#include "llvm/ADT/IntrusiveRefCntPtr.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/StringMap.h"
#include <memory>

namespace llvm {
class MemoryBuffer;
} // end namespace llvm

namespace clang {

/// In-memory cache for modules.
///
/// This is a cache for modules for use across a compilation, sharing state
/// between the CompilerInstances in an implicit modules build.  It must be
/// shared by each CompilerInstance, ASTReader, ASTWriter, and ModuleManager
/// that are coordinating.
///
/// Critically, it ensures that a single process has a consistent view of each
/// PCM.  This is used by \a CompilerInstance when building PCMs to ensure that
/// each \a ModuleManager sees the same files.
///
/// \a finalizeCurrentBuffers() should be called before creating a new user.
/// This locks in the current PCMs, ensuring that no PCM that has already been
/// accessed can be purged, preventing use-after-frees.
class InMemoryModuleCache : public llvm::RefCountedBase<InMemoryModuleCache> {
  struct PCM {
    std::unique_ptr<llvm::MemoryBuffer> Buffer;

    /// Track the timeline of when this was added to the cache.
    unsigned Index;
  };

  /// Cache of buffers.
  llvm::StringMap<PCM> PCMs;

  /// Monotonically increasing index.
  unsigned NextIndex = 0;

  /// Bumped to prevent "older" buffers from being removed.
  unsigned FirstRemovableIndex = 0;

public:
  /// Store the Buffer under the Filename.
  ///
  /// \pre There is not already buffer is not already in the cache.
  /// \return a reference to the buffer as a convenience.
  llvm::MemoryBuffer &addBuffer(llvm::StringRef Filename,
                                std::unique_ptr<llvm::MemoryBuffer> Buffer);

  /// Try to remove a buffer from the cache.
  ///
  /// \return false on success, iff \c !isBufferFinal().
  bool tryToRemoveBuffer(llvm::StringRef Filename);

  /// Get a pointer to the buffer if it exists; else nullptr.
  llvm::MemoryBuffer *lookupBuffer(llvm::StringRef Filename);

  /// Check whether the buffer is final.
  ///
  /// \return true iff \a finalizeCurrentBuffers() has been called since the
  /// buffer was added.  This prevents buffers from being removed.
  bool isBufferFinal(llvm::StringRef Filename);

  /// Finalize the current buffers in the cache.
  ///
  /// Should be called when creating a new user to ensure previous uses aren't
  /// invalidated.
  void finalizeCurrentBuffers();
};

} // end namespace clang

#endif // LLVM_CLANG_SERIALIZATION_INMEMORYMODULECACHE_H
