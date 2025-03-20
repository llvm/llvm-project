//===-- SBLock.h ----------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_API_SBLOCK_H
#define LLDB_API_SBLOCK_H

#include "lldb/API/SBDefines.h"
#include "lldb/lldb-forward.h"
#include <mutex>

namespace lldb_private {
class APILock;
}

namespace lldb {

/// A general-purpose lock in the SB API. The lock can be locked and unlocked.
/// The default constructed lock is unlocked, but generally the lock is locked
/// when it is returned from a class.
class LLDB_API SBLock {
public:
  SBLock();
  SBLock(SBLock &&rhs);
  SBLock &operator=(SBLock &&rhs);
  ~SBLock();

  /// Returns true if this lock has ownership of the underlying mutex.
  bool IsValid() const;

  /// Blocking operation that takes ownership of this lock.
  void Lock() const;

  /// Releases ownership of this lock.
  void Unlock() const;

private:
  // Private constructor used by SBTarget to create the Target API lock.
  // Requires a friend declaration.
  SBLock(lldb::TargetSP target_sp);
  friend class SBTarget;

  SBLock(const SBLock &rhs) = delete;
  const SBLock &operator=(const SBLock &rhs) = delete;

  std::unique_ptr<lldb_private::APILock> m_opaque_up;
};
#endif

} // namespace lldb
