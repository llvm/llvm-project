//===-- SBMutex.h
//----------------------------------------------------------===//
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

namespace lldb {

/// A general-purpose lock in the SB API. The lock can be locked and unlocked.
/// The default constructed lock is unlocked, but generally the lock is locked
/// when it is returned from a class.
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

private:
  // Private constructor used by SBTarget to create the Target API mutex.
  // Requires a friend declaration.
  SBMutex(lldb::TargetSP target_sp);
  friend class SBTarget;

  std::shared_ptr<std::recursive_mutex> m_opaque_sp;
};
#endif

} // namespace lldb
