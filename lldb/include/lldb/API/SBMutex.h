//===-- SBMutex.h ---------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_API_SBMUTEX_H
#define LLDB_API_SBMUTEX_H

#include "lldb/API/SBDefines.h"
#include "lldb/lldb-forward.h"
#include <mutex>

namespace lldb {

class LLDB_API SBMutex {
public:
  SBMutex();
  SBMutex(const SBMutex &rhs);
  const SBMutex &operator=(const SBMutex &rhs);
  ~SBMutex();

  /// Returns true if this lock has ownership of the underlying mutex.
  bool IsValid() const;

  /// Blocking operation that takes ownership of this lock.
  void lock() const;

  /// Releases ownership of this lock.
  void unlock() const;

  /// Tries to lock the mutex. Returns immediately. On successful lock
  /// acquisition returns true, otherwise returns false.
  bool try_lock() const;

private:
  // Private constructor used by SBTarget to create the Target API mutex.
  // Requires a friend declaration.
  SBMutex(lldb::TargetSP target_sp);
  friend class SBTarget;

  std::shared_ptr<std::recursive_mutex> m_opaque_sp;
};

} // namespace lldb

#endif
