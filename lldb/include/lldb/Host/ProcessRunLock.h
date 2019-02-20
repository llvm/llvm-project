//===-- ProcessRunLock.h ----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_ProcessRunLock_h_
#define liblldb_ProcessRunLock_h_

#include <stdint.h>
#include <time.h>

#include "lldb/lldb-defines.h"

//----------------------------------------------------------------------
/// Enumerations for broadcasting.
//----------------------------------------------------------------------
namespace lldb_private {

//----------------------------------------------------------------------
/// @class ProcessRunLock ProcessRunLock.h "lldb/Host/ProcessRunLock.h"
/// A class used to prevent the process from starting while other
/// threads are accessing its data, and prevent access to its data while it is
/// running.
//----------------------------------------------------------------------

class ProcessRunLock {
public:
  ProcessRunLock();
  ~ProcessRunLock();

  bool ReadTryLock();
  bool ReadUnlock();
  bool SetRunning();
  bool TrySetRunning();
  bool SetStopped();

  class ProcessRunLocker {
  public:
    ProcessRunLocker() : m_lock(nullptr) {}

    ~ProcessRunLocker() { Unlock(); }

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

    ProcessRunLock *m_lock;

  private:
    DISALLOW_COPY_AND_ASSIGN(ProcessRunLocker);
  };

protected:
  lldb::rwlock_t m_rwlock;
  bool m_running;

private:
  DISALLOW_COPY_AND_ASSIGN(ProcessRunLock);
};

} // namespace lldb_private

#endif // liblldb_ProcessRunLock_h_
