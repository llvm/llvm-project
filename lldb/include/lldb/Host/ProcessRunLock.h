//===-- ProcessRunLock.h ----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_HOST_PROCESSRUNLOCK_H
#define LLDB_HOST_PROCESSRUNLOCK_H

#include <cassert>
#include <cstdint>
#include <ctime>
#include <mutex>

#include "llvm/ADT/DenseMap.h"

#include "lldb/lldb-defines.h"
#include "lldb/lldb-types.h"

/// Enumerations for broadcasting.
namespace lldb_private {

/// \class ProcessRunLock ProcessRunLock.h "lldb/Host/ProcessRunLock.h"
/// Read/write lock around the process running/stopped state.
///
/// POSIX rwlocks aren't reader-recursive: a thread holding the read
/// lock can deadlock against a pending writer if it re-acquires
/// directly. ProcessRunLocker tracks a per-thread recursion count so
/// re-entry skips the rwlock; the raw primitives are private.

class ProcessRunLock {
public:
  ProcessRunLock();
  ~ProcessRunLock();

  /// Set the process to running. Returns true if the process was stopped.
  /// Return false if the process was running.
  bool SetRunning();

  /// Set the process to stopped. Returns true if the process was running.
  /// Returns false if the process was stopped.
  bool SetStopped();

  /// RAII helper around the read-lock side of ProcessRunLock. Supports
  /// same-thread recursion (see class doc).
  ///
  /// Move-assignment unlocks the destination first, then takes the
  /// source's lock. Cross-thread move of a held locker is fatal — the
  /// same thread that called rdlock must call unlock.
  class ProcessRunLocker {
  public:
    ProcessRunLocker() = default;
    ProcessRunLocker(ProcessRunLocker &&other);
    ProcessRunLocker &operator=(ProcessRunLocker &&other);
    ~ProcessRunLocker() { Unlock(); }

    bool IsLocked() const { return m_lock; }

    /// Try to acquire the read lock. If this thread already holds the
    /// read lock on this ProcessRunLock, the underlying rwlock is bypassed
    /// and the per-instance recursion count for this thread is bumped
    /// instead.
    bool TryLock(ProcessRunLock *lock);

  protected:
    void Unlock();

    ProcessRunLock *m_lock = nullptr;
    uint64_t m_thread = 0;

  private:
    ProcessRunLocker(const ProcessRunLocker &) = delete;
    const ProcessRunLocker &operator=(const ProcessRunLocker &) = delete;
  };

protected:
  lldb::rwlock_t m_rwlock;
  bool m_running = false;

private:
  ProcessRunLock(const ProcessRunLock &) = delete;
  const ProcessRunLock &operator=(const ProcessRunLock &) = delete;

  bool ReadTryLock();
  bool ReadUnlock();
  friend class ProcessRunLocker;

  std::mutex m_recursion_mutex;
  llvm::DenseMap<uint64_t, uint32_t> m_recursion;
};

} // namespace lldb_private

#endif // LLDB_HOST_PROCESSRUNLOCK_H
