//===-- ProcessRunLock.h ----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_HOST_PROCESSRUNLOCK_H
#define LLDB_HOST_PROCESSRUNLOCK_H

#include <cstdint>
#include <ctime>

#include "lldb/lldb-defines.h"

/// Enumerations for broadcasting.
namespace lldb_private {

/// \class ProcessRunLock ProcessRunLock.h "lldb/Host/ProcessRunLock.h"
/// A class used to prevent the process from starting while other
/// threads are accessing its data, and prevent access to its data while it is
/// running.

class ProcessRunLock {
public:
  ProcessRunLock();
  ~ProcessRunLock();

  bool ReadTryLock();
  bool ReadUnlock();

  /// Set the process to running. Returns true if the process was stopped.
  /// Return false if the process was running.
  bool SetRunning();

  /// Set the process to stopped. Returns true if the process was running.
  /// Returns false if the process was stopped.
  bool SetStopped();

  class ProcessRunLocker {
  public:
    ProcessRunLocker() = default;
    ProcessRunLocker(ProcessRunLocker &&other) : m_lock(other.m_lock) {
      other.m_lock = nullptr;
    }
    ProcessRunLocker &operator=(ProcessRunLocker &&other) {
      if (this != &other) {
        Unlock();
        m_lock = other.m_lock;
        other.m_lock = nullptr;
      }
      return *this;
    }

    ~ProcessRunLocker() { Unlock(); }

    bool IsLocked() const { return m_lock; }

    // Try to lock the read lock, but only do so if there are no writers.
    bool TryLock(ProcessRunLock *lock) {
      if (m_lock) {
        if (m_lock == lock)
          return true; // We already have this lock locked
        else
          Unlock();
      }
      if (lock) {
        if (lock->ReadTryLock()) {
          m_lock = lock;
          return true;
        }
      }
      return false;
    }

  protected:
    void Unlock() {
      if (m_lock) {
        m_lock->ReadUnlock();
        m_lock = nullptr;
      }
    }

    ProcessRunLock *m_lock = nullptr;

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
};

} // namespace lldb_private

#endif // LLDB_HOST_PROCESSRUNLOCK_H
