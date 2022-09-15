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

class TypeSystemSwiftTypeRefForExpressions;
class ExecutionContext;
class ExecutionContextRef;

/// This is like llvm::sys::SmartRWMutex<true>, but with a try_lock method.
///
/// FIXME: Replace this with a C++14 shared_timed_mutex or a C++17
/// share_mutex as soon as possible.
class SharedMutex {
  llvm::sys::SmartRWMutex<true> m_impl;
  unsigned m_readers = 0;
  std::mutex m_reader_mutex;
public:
  SharedMutex() = default;
  SharedMutex(const SharedMutex &) = delete;
  SharedMutex &operator=(const SharedMutex &) = delete;
  bool lock_shared() {
    std::lock_guard<std::mutex> lock(m_reader_mutex);
    ++m_readers;
    return m_impl.lock_shared();
  }
  bool unlock_shared() {
    std::lock_guard<std::mutex> lock(m_reader_mutex);
    assert(m_readers > 0);
    --m_readers;
    return m_impl.unlock_shared();
  }
  bool try_lock() {
    std::lock_guard<std::mutex> lock(m_reader_mutex);
    return m_readers ? false : m_impl.lock();
  }
  bool unlock() { return m_impl.unlock(); }
};

/// RAII acquisition of a reader lock.
struct ScopedSharedMutexReader {
  SharedMutex *m_mutex;
  explicit ScopedSharedMutexReader(SharedMutex *m) : m_mutex(m) {
    if (m_mutex)
      m_mutex->lock_shared();
  }

  ScopedSharedMutexReader(const ScopedSharedMutexReader &copy)
      : m_mutex(copy.m_mutex) {
    if (m_mutex)
      m_mutex->lock_shared();
  }

  ~ScopedSharedMutexReader() {
    if (m_mutex)
      m_mutex->unlock_shared();
  }
};

/// A scratch Swift context pointer and its reader lock.
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
/// GetSwiftScratchContext() will create a fresh global context,
/// or even separate scratch contexts for each lldb::Module. But it
/// will only do this if no client holds on to a read lock on \b
/// m_scratch_typesystem_lock.
class SwiftScratchContextReader : ScopedSharedMutexReader {
  TypeSystemSwiftTypeRefForExpressions *m_ptr;

public:
  SwiftScratchContextReader(SharedMutex &mutex,
                            TypeSystemSwiftTypeRefForExpressions &ctx)
      : ScopedSharedMutexReader(&mutex), m_ptr(&ctx) {
    assert(m_ptr && "invalid context");
  }

  TypeSystemSwiftTypeRefForExpressions *get() {
    assert(m_ptr && "invalid context");
    return m_ptr;
  }

  TypeSystemSwiftTypeRefForExpressions *operator->() { return get(); }
  TypeSystemSwiftTypeRefForExpressions &operator*() { return *get(); }
};

/// An RAII object that just acquires the reader lock.
struct SwiftScratchContextLock : ScopedSharedMutexReader {
  SwiftScratchContextLock(const ExecutionContextRef *exe_ctx_ref);
  SwiftScratchContextLock(const ExecutionContext *exe_ctx);
};

} // namespace lldb_private
#endif
