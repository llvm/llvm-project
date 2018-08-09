//===-- SwiftASTContextReader.h ---------------------------------*- C++ -*-===//
//
// This source file is part of the Swift.org open source project
//
// Copyright (c) 2018 Apple Inc. and the Swift project authors
// Licensed under Apache License v2.0 with Runtime Library Exception
//
// See https://swift.org/LICENSE.txt for license information
// See https://swift.org/CONTRIBUTORS.txt for the list of Swift project authors
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_SwiftASTContextReader_h_
#define liblldb_SwiftASTContextReader_h_

#include "llvm/Support/RWMutex.h"

namespace lldb_private {

class SwiftASTContext;
class ExecutionContext;
class ExecutionContextRef;

/// This is like llvm::sys::SmartRWMutex<false>, but with a try_lock method.
///
/// FIXME: Replace this with a C++14 shared_timed_mutex or a C++17
/// share_mutex as soon as possible. This implementation still allows
/// for a race condition in \c try_lock() between the checking of
/// m_readers and the lock acquisition.
class SharedMutex {
  llvm::sys::RWMutexImpl m_impl;
  unsigned m_readers = 0;
public:
  SharedMutex() = default;
  SharedMutex(const SharedMutex &) = delete;
  SharedMutex &operator=(const SharedMutex &) = delete;
  bool lock_shared() { ++m_readers; return m_impl.reader_acquire(); }
  bool unlock_shared() { --m_readers; return m_impl.reader_release(); }
  bool try_lock() { return m_readers ? false : m_impl.writer_acquire(); }
  bool unlock() { return m_impl.writer_release(); }
};

/// RAII acquisition of a reader lock.
struct ScopedSharedMutexReader {
  SharedMutex *m_mutex;
  ScopedSharedMutexReader(const ScopedSharedMutexReader&) = default;
  explicit ScopedSharedMutexReader(SharedMutex *m) : m_mutex(m) {
    if (m_mutex)
      m_mutex->lock_shared();
  }

  ~ScopedSharedMutexReader() {
    if (m_mutex)
      m_mutex->unlock_shared();
  }
};

/// A scratch Swift AST context pointer and its reader lock.
/// The Swift scratch context may need to be replaced when it gets corrupted,
/// for example due to incompatible ClangImporter options. This locking
/// mechanism guarantees that this won't happen while a client is using the
/// context.
///
/// In Swift there are three use-cases for ASTContexts with
/// different requirements and guarantees.
///
/// - Module ASTContexts are used for the static type system. They
///   are created once for each lldb::Module and live forever.
///
/// - Scratch AST Contexts are used for expressions
///   (thus far everything is like in the Clang language support)
///
/// - Scratch AST Contexts are also used to express the results of
///    any dynamic type resolution done by RemoteAST or Archetype
///    binding.
///
/// Because expressions and dynamic type resolution may trigger the
/// import of another module, the scratch context may become
/// unusable. When a scratch context is in a fatal error state,
/// GetScratchSwiftASTContext() will create a fresh global context,
/// or even separate scratch contexts for each lldb::Module. But it
/// will only do this if no client holds on to a read lock on \b
/// m_scratch_typesystem_lock.
struct SwiftASTContextReader : ScopedSharedMutexReader {
  SwiftASTContext *m_ptr = nullptr;
  SwiftASTContextReader() : ScopedSharedMutexReader(nullptr) {}
  SwiftASTContextReader(SharedMutex &mutex, SwiftASTContext *ctx)
      : ScopedSharedMutexReader(&mutex), m_ptr(ctx) {}
  SwiftASTContextReader(const SwiftASTContextReader &copy)
      : ScopedSharedMutexReader(copy.m_mutex), m_ptr(copy.m_ptr) {}
  SwiftASTContext *get() { return m_ptr; }
  operator bool() const { return m_ptr; }
  SwiftASTContext *operator->() { return m_ptr; }
  SwiftASTContext &operator*() { return *m_ptr; }
};

/// An RAII object that just acquires the reader lock.
struct SwiftASTContextLock : ScopedSharedMutexReader {
  SwiftASTContextLock(const ExecutionContextRef *exe_ctx_ref);
  SwiftASTContextLock(const ExecutionContext *exe_ctx);
};

} // namespace lldb_private
#endif
