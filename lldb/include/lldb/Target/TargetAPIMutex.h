//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_TARGET_TARGETAPIMUTEX_H
#define LLDB_TARGET_TARGETAPIMUTEX_H

#include "lldb/lldb-forward.h"
#include <memory>
#include <mutex>

namespace lldb_private {

/// A Lockable handle over a Target's API mutex, returned by
/// Target::GetAPIMutex() and backing the public lldb::SBMutex.
///
/// Behaves like std::recursive_mutex: lock()/try_lock()/unlock() drive
/// the actual synchronization, with the same contract (unlock() without
/// a matching successful lock()/try_lock() is caller error). It carries
/// no RAII of its own; wrap it in std::lock_guard<TargetAPIMutex> or
/// std::unique_lock<TargetAPIMutex> for scope-based locking, exactly as
/// with any other Lockable.
///
/// A handle may be constructed on one thread and then locked/unlocked
/// on a different one, so lock()/try_lock() (re-)resolve which real
/// mutex to use fresh on every call, rather than caching a single
/// resolution for the handle's lifetime. The matching unlock() replays
/// the exact resolution that call produced, rather than re-resolving,
/// so the calling thread's policy at unlock() time can't cause it to
/// release the wrong mutex (or fail to release the one it actually
/// holds).
///
/// Default-constructed (or moved-from) handles are a genuine no-op: no
/// synchronization primitive is touched at all.
class TargetAPIMutex {
public:
  TargetAPIMutex() = default;
  explicit TargetAPIMutex(lldb::TargetSP target_sp)
      : m_target_sp(std::move(target_sp)) {}

  TargetAPIMutex(TargetAPIMutex &&other) noexcept = default;
  TargetAPIMutex &operator=(TargetAPIMutex &&other) noexcept = default;

  TargetAPIMutex(const TargetAPIMutex &) = delete;
  TargetAPIMutex &operator=(const TargetAPIMutex &) = delete;

  void lock();
  bool try_lock();
  void unlock() {
    if (m_mutex)
      m_mutex->unlock();
  }

private:
  /// An aliasing shared_ptr into m_target_sp's own mutex, resolved fresh
  /// on every lock()/try_lock() call. Shares m_target_sp's control block
  /// (keeping the Target alive) while pointing at the mutex living inside
  /// it. Null when this handle is a genuine no-op.
  std::shared_ptr<std::recursive_mutex> m_mutex;
  lldb::TargetSP m_target_sp;
};

} // namespace lldb_private

#endif // LLDB_TARGET_TARGETAPIMUTEX_H
