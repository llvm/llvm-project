//===-- LockFileBase.h ------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_Host_LockFileBase_h_
#define liblldb_Host_LockFileBase_h_

#include "lldb/Utility/Error.h"

#include <functional>

namespace lldb_private {

class LockFileBase {
public:
  virtual ~LockFileBase() = default;

  bool IsLocked() const;

  Error WriteLock(const uint64_t start, const uint64_t len);
  Error TryWriteLock(const uint64_t start, const uint64_t len);

  Error ReadLock(const uint64_t start, const uint64_t len);
  Error TryReadLock(const uint64_t start, const uint64_t len);

  Error Unlock();

protected:
  using Locker = std::function<Error(const uint64_t, const uint64_t)>;

  LockFileBase(int fd);

  virtual bool IsValidFile() const;

  virtual Error DoWriteLock(const uint64_t start, const uint64_t len) = 0;
  virtual Error DoTryWriteLock(const uint64_t start, const uint64_t len) = 0;

  virtual Error DoReadLock(const uint64_t start, const uint64_t len) = 0;
  virtual Error DoTryReadLock(const uint64_t start, const uint64_t len) = 0;

  virtual Error DoUnlock() = 0;

  Error DoLock(const Locker &locker, const uint64_t start, const uint64_t len);

  int m_fd; // not owned.
  bool m_locked;
  uint64_t m_start;
  uint64_t m_len;
};
}

#endif
