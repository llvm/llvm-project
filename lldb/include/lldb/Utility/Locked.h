//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_UTILITY_LOCKED_H
#define LLDB_UTILITY_LOCKED_H

#include "llvm/Support/RWMutex.h"

#include <cassert>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <type_traits>
#include <utility>

namespace lldb_private {

namespace detail {
/// Common pointer-like accessors shared by `Locked` and `SharedLocked`.
template <typename Derived, typename PtrT> class LockedAccessors {
public:
  auto operator->() const { return Raw(); }
  decltype(auto) operator*() const { return *Self()->m_ptr; }
  auto get() const { return Raw(); }
  explicit operator bool() const { return Raw() != nullptr; }

private:
  const Derived *Self() const { return static_cast<const Derived *>(this); }

  auto Raw() const {
    if constexpr (std::is_pointer_v<PtrT>)
      return Self()->m_ptr;
    else
      return Self()->m_ptr.get();
  }
};
} // namespace detail

/// A move-only RAII handle that pairs a pointer-like value with an exclusive
/// lock on a caller-supplied mutex. While the handle is alive the borrowed
/// pointer is serialized against other threads that go through the same
/// mutex.
///
/// `PtrT` is the pointer-like value: a raw pointer (`T*`),
/// `std::shared_ptr<T>`, or `std::unique_ptr<T>`. `Mutex` may be any type
/// that satisfies `Lockable` — `std::mutex`, `std::recursive_mutex`,
/// `std::shared_mutex`, or `llvm::sys::RWMutex` all work. Use the
/// `LockedPtr`, `LockedSP`, `LockedUP` aliases for the common combinations.
template <typename PtrT, typename Mutex = std::recursive_mutex>
class Locked : public detail::LockedAccessors<Locked<PtrT, Mutex>, PtrT> {
  friend class detail::LockedAccessors<Locked<PtrT, Mutex>, PtrT>;

public:
  using mutex_type = Mutex;
  using lock_type = std::unique_lock<Mutex>;

  Locked() = default;
  Locked(mutex_type &m, PtrT p) : m_lock(m), m_ptr(std::move(p)) {}
  Locked(lock_type lock, PtrT p)
      : m_lock(std::move(lock)), m_ptr(std::move(p)) {
    assert(m_lock.owns_lock() && "Locked requires an owning lock");
  }

  Locked(Locked &&other)
      : m_lock(std::move(other.m_lock)),
        m_ptr(std::exchange(other.m_ptr, PtrT{})) {}
  Locked &operator=(Locked &&other) {
    m_lock = std::move(other.m_lock);
    m_ptr = std::exchange(other.m_ptr, PtrT{});
    return *this;
  }
  Locked(const Locked &) = delete;
  Locked &operator=(const Locked &) = delete;

private:
  lock_type m_lock;
  PtrT m_ptr{};
};

/// A copyable RAII handle that pairs a pointer-like value with a shared
/// (reader) lock on a caller-supplied mutex. Copies share the same
/// underlying reader lock through reference counting; the lock is released
/// when the last copy is destroyed. This makes a `SharedLocked` cheap to
/// pass through code paths that branch or fan out without each leaf having
/// to re-acquire.
///
/// The borrowed pointer is `const`-qualified so callers cannot mutate the
/// pointee while holding only a reader's lock. `Mutex` must satisfy
/// `SharedLockable` — `llvm::sys::RWMutex` (the LLDB convention) or
/// `std::shared_mutex`. Use the `SharedLockedPtr`, `SharedLockedSP`,
/// `SharedLockedUP` aliases for the common combinations.
template <typename PtrT, typename Mutex = llvm::sys::RWMutex>
class SharedLocked
    : public detail::LockedAccessors<SharedLocked<PtrT, Mutex>, PtrT> {
  friend class detail::LockedAccessors<SharedLocked<PtrT, Mutex>, PtrT>;

  // The class is intended to be instantiated only via the
  // SharedLockedPtr/SP/UP aliases, which all bake in `const`. Enforcing it
  // here catches direct uses that would otherwise hand readers a mutable
  // pointer.
  static constexpr bool PointeeIsConst = []() {
    if constexpr (std::is_pointer_v<PtrT>)
      return std::is_const_v<std::remove_pointer_t<PtrT>>;
    else
      return std::is_const_v<typename PtrT::element_type>;
  }();
  static_assert(PointeeIsConst,
                "SharedLocked requires a pointer to a const-qualified type; "
                "use the SharedLockedPtr/SP/UP aliases.");

public:
  using mutex_type = Mutex;
  using lock_type = std::shared_lock<Mutex>;

  SharedLocked() = default;
  SharedLocked(mutex_type &m, PtrT p)
      : m_lock(std::make_shared<lock_type>(m)), m_ptr(std::move(p)) {}
  SharedLocked(lock_type lock, PtrT p)
      : m_lock(std::make_shared<lock_type>(std::move(lock))),
        m_ptr(std::move(p)) {
    assert(m_lock->owns_lock() && "SharedLocked requires an owning lock");
  }

  SharedLocked(const SharedLocked &) = default;
  SharedLocked &operator=(const SharedLocked &) = default;
  SharedLocked(SharedLocked &&other)
      : m_lock(std::move(other.m_lock)),
        m_ptr(std::exchange(other.m_ptr, PtrT{})) {}
  SharedLocked &operator=(SharedLocked &&other) {
    m_lock = std::move(other.m_lock);
    m_ptr = std::exchange(other.m_ptr, PtrT{});
    return *this;
  }

private:
  std::shared_ptr<lock_type> m_lock;
  PtrT m_ptr{};
};

/// Exclusive (write) access aliases. The default mutex is
/// `std::recursive_mutex` to match the existing LLDB synchronization style.
/// @{
template <typename T, typename Mutex = std::recursive_mutex>
using LockedPtr = Locked<T *, Mutex>;

template <typename T, typename Mutex = std::recursive_mutex>
using LockedSP = Locked<std::shared_ptr<T>, Mutex>;

template <typename T, typename Mutex = std::recursive_mutex>
using LockedUP = Locked<std::unique_ptr<T>, Mutex>;
/// @}

/// Shared (read) access aliases. The default mutex is `llvm::sys::RWMutex`,
/// the LLDB convention for read/write locks.
/// @{
template <typename T, typename Mutex = llvm::sys::RWMutex>
using SharedLockedPtr = SharedLocked<const T *, Mutex>;

template <typename T, typename Mutex = llvm::sys::RWMutex>
using SharedLockedSP = SharedLocked<std::shared_ptr<const T>, Mutex>;

template <typename T, typename Mutex = llvm::sys::RWMutex>
using SharedLockedUP = SharedLocked<std::unique_ptr<const T>, Mutex>;
/// @}

} // namespace lldb_private

#endif // LLDB_UTILITY_LOCKED_H
