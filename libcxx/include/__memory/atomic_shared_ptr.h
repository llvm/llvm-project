// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___MEMORY_ATOMIC_SHARED_PTR_H
#define _LIBCPP___MEMORY_ATOMIC_SHARED_PTR_H

#include <__atomic/atomic_sync_lite.h>
#include <__atomic/check_memory_order.h>
#include <__atomic/memory_order.h>
#include <__atomic/support.h>
#include <__config>
#include <__cstddef/nullptr_t.h>
#include <__memory/shared_count.h>
#include <__utility/move.h>

#include <cstdint>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

#if defined(__SANITIZE_THREAD__) || (defined(__has_feature) && __has_feature(thread_sanitizer))
#  if __has_include(<sanitizer/tsan_interface.h>)
#    include <sanitizer/tsan_interface.h>
#    define _LIBCPP_ATOMIC_SHARED_PTR_TSAN 1
#  endif
#endif

_LIBCPP_PUSH_MACROS
#include <__undef_macros>

// TSAN annotations model the lock-bit protocol on __ctrl_.
#if defined(_LIBCPP_ATOMIC_SHARED_PTR_TSAN)
#  define _LIBCPP_ATOMIC_SP_TSAN_PRE_LOCK(addr)                                                                        \
    ::__tsan_mutex_pre_lock(reinterpret_cast<void*>(const_cast<__cxx_atomic_impl<uintptr_t>*>(addr)), 0)
#  define _LIBCPP_ATOMIC_SP_TSAN_POST_LOCK(addr)                                                                       \
    ::__tsan_mutex_post_lock(reinterpret_cast<void*>(const_cast<__cxx_atomic_impl<uintptr_t>*>(addr)), 0, 0)
#  define _LIBCPP_ATOMIC_SP_TSAN_PRE_UNLOCK(addr)                                                                      \
    ::__tsan_mutex_pre_unlock(reinterpret_cast<void*>(const_cast<__cxx_atomic_impl<uintptr_t>*>(addr)), 0)
#  define _LIBCPP_ATOMIC_SP_TSAN_POST_UNLOCK(addr)                                                                     \
    ::__tsan_mutex_post_unlock(reinterpret_cast<void*>(const_cast<__cxx_atomic_impl<uintptr_t>*>(addr)), 0)
#else
#  define _LIBCPP_ATOMIC_SP_TSAN_PRE_LOCK(addr) ((void)(addr))
#  define _LIBCPP_ATOMIC_SP_TSAN_POST_LOCK(addr) ((void)(addr))
#  define _LIBCPP_ATOMIC_SP_TSAN_PRE_UNLOCK(addr) ((void)(addr))
#  define _LIBCPP_ATOMIC_SP_TSAN_POST_UNLOCK(addr) ((void)(addr))
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

#if _LIBCPP_STD_VER >= 20 && _LIBCPP_HAS_THREADS && _LIBCPP_HAS_ATOMIC_HEADER

template <class _Tp>
class shared_ptr;

template <class _Tp>
class weak_ptr;

template <class _Tp>
struct atomic;

// Split state into pointer word and control word.
// The control word stores control-block pointer plus lock/notify bits.
struct __atomic_smart_ptr_storage {
  static constexpr uintptr_t __lock_bit_   = uintptr_t{1};
  static constexpr uintptr_t __notify_bit_ = uintptr_t{2};
  static constexpr uintptr_t __ptr_mask_   = ~(__lock_bit_ | __notify_bit_);

  _LIBCPP_HIDE_FROM_ABI static uintptr_t __encode(__shared_weak_count* __ctrl, uintptr_t __bits) _NOEXCEPT {
    return (reinterpret_cast<uintptr_t>(__ctrl) & __ptr_mask_) | (__bits & ~__ptr_mask_);
  }

  _LIBCPP_HIDE_FROM_ABI static __shared_weak_count* __decode(uintptr_t __word) _NOEXCEPT {
    return reinterpret_cast<__shared_weak_count*>(__word & __ptr_mask_);
  }

  _LIBCPP_HIDE_FROM_ABI static bool __has_lock(uintptr_t __word) _NOEXCEPT { return (__word & __lock_bit_) != 0; }
  _LIBCPP_HIDE_FROM_ABI static bool __has_notify(uintptr_t __word) _NOEXCEPT { return (__word & __notify_bit_) != 0; }
};

_LIBCPP_HIDE_FROM_ABI inline void __atomic_smart_ptr_notify_one(const void* __address) _NOEXCEPT {
#  if _LIBCPP_AVAILABILITY_HAS_NEW_SYNC
  std::__atomic_notify_one_global_table(__address);
#  else
  std::__cxx_atomic_notify_one(reinterpret_cast<void const volatile*>(__address));
#  endif
}

_LIBCPP_HIDE_FROM_ABI inline void __atomic_smart_ptr_notify_all(const void* __address) _NOEXCEPT {
#  if _LIBCPP_AVAILABILITY_HAS_NEW_SYNC
  std::__atomic_notify_all_global_table(__address);
#  else
  std::__cxx_atomic_notify_all(reinterpret_cast<void const volatile*>(__address));
#  endif
}

template <class _Poll>
_LIBCPP_HIDE_FROM_ABI inline void __atomic_smart_ptr_wait_on_address(const void* __address, _Poll&& __poll) _NOEXCEPT {
  while (!__poll()) {
#  if _LIBCPP_AVAILABILITY_HAS_NEW_SYNC
    auto __monitor_value = std::__atomic_monitor_global(__address);
    if (__poll())
      return;
    std::__atomic_wait_global_table(__address, __monitor_value);
#  else
    void const volatile* __volatile_address = reinterpret_cast<void const volatile*>(__address);
    auto __monitor_value                    = std::__libcpp_atomic_monitor(__volatile_address);
    if (__poll())
      return;
    std::__libcpp_atomic_wait(__volatile_address, __monitor_value);
#  endif
  }
}

template <class _Element>
struct __atomic_smart_ptr_fields {
  mutable __cxx_atomic_impl<_Element*> __ptr_;
  mutable __cxx_atomic_impl<uintptr_t> __ctrl_;

  _LIBCPP_HIDE_FROM_ABI __atomic_smart_ptr_fields(_Element* __p, __shared_weak_count* __c) _NOEXCEPT
      : __ptr_(__p),
        __ctrl_(__atomic_smart_ptr_storage::__encode(__c, 0)) {}

  _LIBCPP_HIDE_FROM_ABI const void* __ctrl_address() const _NOEXCEPT {
    return static_cast<const void*>(__builtin_addressof(__ctrl_));
  }

  // Acquire lock bit on __ctrl_. Contended path sets notify bit and waits.
  _LIBCPP_HIDE_FROM_ABI void __lock() const _NOEXCEPT {
    _LIBCPP_ATOMIC_SP_TSAN_PRE_LOCK(&__ctrl_);
    uintptr_t __expected = std::__cxx_atomic_load(__builtin_addressof(__ctrl_), memory_order_relaxed);
    for (;;) {
      if (!__atomic_smart_ptr_storage::__has_lock(__expected)) {
        uintptr_t __desired = __expected | __atomic_smart_ptr_storage::__lock_bit_;
        if (std::__cxx_atomic_compare_exchange_weak(
                __builtin_addressof(__ctrl_),
                __builtin_addressof(__expected),
                __desired,
                memory_order_acquire,
                memory_order_relaxed)) {
          _LIBCPP_ATOMIC_SP_TSAN_POST_LOCK(&__ctrl_);
          return;
        }
        continue;
      }

      uintptr_t __with_notify = __expected | __atomic_smart_ptr_storage::__notify_bit_;
      if (!__atomic_smart_ptr_storage::__has_notify(__expected)) {
        if (!std::__cxx_atomic_compare_exchange_weak(
                __builtin_addressof(__ctrl_),
                __builtin_addressof(__expected),
                __with_notify,
                memory_order_relaxed,
                memory_order_relaxed))
          continue;
        __expected = __with_notify;
      }

      std::__atomic_smart_ptr_wait_on_address(__ctrl_address(), [&] {
        __expected = std::__cxx_atomic_load(__builtin_addressof(__ctrl_), memory_order_relaxed);
        return !__atomic_smart_ptr_storage::__has_lock(__expected);
      });
    }
  }

  // Publish new control pointer, clear bits, and notify waiters if needed.
  _LIBCPP_HIDE_FROM_ABI void __unlock(__shared_weak_count* __ctrl_to_publish) const _NOEXCEPT {
    _LIBCPP_ATOMIC_SP_TSAN_PRE_UNLOCK(&__ctrl_);
    uintptr_t __new_word = __atomic_smart_ptr_storage::__encode(__ctrl_to_publish, 0);
    uintptr_t __previous = std::__cxx_atomic_exchange(__builtin_addressof(__ctrl_), __new_word, memory_order_release);
    if (__atomic_smart_ptr_storage::__has_notify(__previous))
      std::__atomic_smart_ptr_notify_all(__ctrl_address());
    _LIBCPP_ATOMIC_SP_TSAN_POST_UNLOCK(&__ctrl_);
  }
};

// [util.smartptr.atomic.shared]: same stored pointer and same ownership, or both empty.
template <class _Element>
_LIBCPP_HIDE_FROM_ABI inline bool __atomic_smart_ptr_equivalent(
    _Element* __ptr,
    __shared_weak_count* __ctrl,
    _Element* __expected_ptr,
    __shared_weak_count* __expected_ctrl) _NOEXCEPT {
  if (__ctrl == nullptr && __expected_ctrl == nullptr)
    return true;
  return __ptr == __expected_ptr && __ctrl == __expected_ctrl;
}

template <class _Tp>
struct atomic<shared_ptr<_Tp>> {
  using value_type = shared_ptr<_Tp>;

  static constexpr bool is_always_lock_free = false;

  _LIBCPP_HIDE_FROM_ABI atomic() _NOEXCEPT : __fields_(nullptr, nullptr) {}
  _LIBCPP_HIDE_FROM_ABI constexpr atomic(nullptr_t) _NOEXCEPT : __fields_(nullptr, nullptr) {}
  _LIBCPP_HIDE_FROM_ABI atomic(shared_ptr<_Tp> __desired) _NOEXCEPT : __fields_(__desired.__ptr_, __desired.__cntrl_) {
    __desired.__ptr_   = nullptr;
    __desired.__cntrl_ = nullptr;
  }

  atomic(const atomic&)            = delete;
  atomic& operator=(const atomic&) = delete;

  _LIBCPP_HIDE_FROM_ABI ~atomic() {
    if (auto* __c = __atomic_smart_ptr_storage::__decode(
            std::__cxx_atomic_load(__builtin_addressof(__fields_.__ctrl_), memory_order_relaxed)))
      __c->__release_shared();
  }

  _LIBCPP_HIDE_FROM_ABI bool is_lock_free() const _NOEXCEPT { return false; }

  _LIBCPP_HIDE_FROM_ABI void operator=(shared_ptr<_Tp> __desired) _NOEXCEPT { store(std::move(__desired)); }
  _LIBCPP_HIDE_FROM_ABI void operator=(nullptr_t) _NOEXCEPT { store(nullptr); }
  _LIBCPP_HIDE_FROM_ABI operator shared_ptr<_Tp>() const _NOEXCEPT { return load(); }

  _LIBCPP_HIDE_FROM_ABI void store(shared_ptr<_Tp> __desired, memory_order __m = memory_order_seq_cst) _NOEXCEPT
      _LIBCPP_CHECK_STORE_MEMORY_ORDER(__m) {
    (void)__m;
    _Tp* __desired_ptr               = __desired.__ptr_;
    __shared_weak_count* __desired_c = __desired.__cntrl_;
    __desired.__ptr_                 = nullptr;
    __desired.__cntrl_               = nullptr;

    __fields_.__lock();
    __shared_weak_count* __old_c = __atomic_smart_ptr_storage::__decode(
        std::__cxx_atomic_load(__builtin_addressof(__fields_.__ctrl_), memory_order_relaxed));
    std::__cxx_atomic_store(__builtin_addressof(__fields_.__ptr_), __desired_ptr, memory_order_relaxed);
    __fields_.__unlock(__desired_c);

    if (__old_c)
      __old_c->__release_shared();
  }

  _LIBCPP_HIDE_FROM_ABI shared_ptr<_Tp> load(memory_order __m = memory_order_seq_cst) const _NOEXCEPT
      _LIBCPP_CHECK_LOAD_MEMORY_ORDER(__m) {
    (void)__m;
    __fields_.__lock();
    _Tp* __ptr               = std::__cxx_atomic_load(__builtin_addressof(__fields_.__ptr_), memory_order_relaxed);
    __shared_weak_count* __c = __atomic_smart_ptr_storage::__decode(
        std::__cxx_atomic_load(__builtin_addressof(__fields_.__ctrl_), memory_order_relaxed));
    if (__c)
      __c->__add_shared();
    __fields_.__unlock(__c);
    return shared_ptr<_Tp>::__create_with_control_block(__ptr, __c);
  }

  _LIBCPP_HIDE_FROM_ABI shared_ptr<_Tp>
  exchange(shared_ptr<_Tp> __desired, memory_order __m = memory_order_seq_cst) _NOEXCEPT {
    (void)__m;
    _Tp* __desired_ptr               = __desired.__ptr_;
    __shared_weak_count* __desired_c = __desired.__cntrl_;
    __desired.__ptr_                 = nullptr;
    __desired.__cntrl_               = nullptr;

    __fields_.__lock();
    _Tp* __old_ptr               = std::__cxx_atomic_load(__builtin_addressof(__fields_.__ptr_), memory_order_relaxed);
    __shared_weak_count* __old_c = __atomic_smart_ptr_storage::__decode(
        std::__cxx_atomic_load(__builtin_addressof(__fields_.__ctrl_), memory_order_relaxed));
    std::__cxx_atomic_store(__builtin_addressof(__fields_.__ptr_), __desired_ptr, memory_order_relaxed);
    __fields_.__unlock(__desired_c);

    return shared_ptr<_Tp>::__create_with_control_block(__old_ptr, __old_c);
  }

  _LIBCPP_HIDE_FROM_ABI bool compare_exchange_strong(
      shared_ptr<_Tp>& __expected, shared_ptr<_Tp> __desired, memory_order __success, memory_order __failure) _NOEXCEPT
      _LIBCPP_CHECK_EXCHANGE_MEMORY_ORDER(__success, __failure) {
    (void)__success;
    (void)__failure;
    __fields_.__lock();
    _Tp* __cur_ptr               = std::__cxx_atomic_load(__builtin_addressof(__fields_.__ptr_), memory_order_relaxed);
    __shared_weak_count* __cur_c = __atomic_smart_ptr_storage::__decode(
        std::__cxx_atomic_load(__builtin_addressof(__fields_.__ctrl_), memory_order_relaxed));

    if (__atomic_smart_ptr_equivalent(__cur_ptr, __cur_c, __expected.__ptr_, __expected.__cntrl_)) {
      _Tp* __desired_ptr               = __desired.__ptr_;
      __shared_weak_count* __desired_c = __desired.__cntrl_;
      __desired.__ptr_                 = nullptr;
      __desired.__cntrl_               = nullptr;

      std::__cxx_atomic_store(__builtin_addressof(__fields_.__ptr_), __desired_ptr, memory_order_relaxed);
      __fields_.__unlock(__desired_c);
      if (__cur_c)
        __cur_c->__release_shared();
      return true;
    }

    if (__cur_c)
      __cur_c->__add_shared();
    __fields_.__unlock(__cur_c);
    __expected = shared_ptr<_Tp>::__create_with_control_block(__cur_ptr, __cur_c);
    return false;
  }

  _LIBCPP_HIDE_FROM_ABI bool compare_exchange_strong(
      shared_ptr<_Tp>& __expected, shared_ptr<_Tp> __desired, memory_order __m = memory_order_seq_cst) _NOEXCEPT {
    return compare_exchange_strong(__expected, std::move(__desired), __m, std::__to_failure_order(__m));
  }

  _LIBCPP_HIDE_FROM_ABI bool compare_exchange_weak(
      shared_ptr<_Tp>& __expected, shared_ptr<_Tp> __desired, memory_order __success, memory_order __failure) _NOEXCEPT
      _LIBCPP_CHECK_EXCHANGE_MEMORY_ORDER(__success, __failure) {
    return compare_exchange_strong(__expected, std::move(__desired), __success, __failure);
  }

  _LIBCPP_HIDE_FROM_ABI bool compare_exchange_weak(
      shared_ptr<_Tp>& __expected, shared_ptr<_Tp> __desired, memory_order __m = memory_order_seq_cst) _NOEXCEPT {
    return compare_exchange_strong(__expected, std::move(__desired), __m, std::__to_failure_order(__m));
  }

  // Wait until the stored value is not equivalent to __old.
  // __ctrl_ is the wait address; pointer changes are published with control updates.
  _LIBCPP_HIDE_FROM_ABI void wait(shared_ptr<_Tp> __old, memory_order __m = memory_order_seq_cst) const _NOEXCEPT
      _LIBCPP_CHECK_WAIT_MEMORY_ORDER(__m) {
    _Tp* __old_ptr               = __old.__ptr_;
    __shared_weak_count* __old_c = __old.__cntrl_;

    std::__atomic_smart_ptr_wait_on_address(__fields_.__ctrl_address(), [&] {
      uintptr_t __word             = std::__cxx_atomic_load(__builtin_addressof(__fields_.__ctrl_), __m);
      __shared_weak_count* __cur_c = __atomic_smart_ptr_storage::__decode(__word);
      if (__cur_c != __old_c)
        return true;
      _Tp* __cur_ptr = std::__cxx_atomic_load(__builtin_addressof(__fields_.__ptr_), __m);
      return !__atomic_smart_ptr_equivalent(__cur_ptr, __cur_c, __old_ptr, __old_c);
    });
  }

  _LIBCPP_HIDE_FROM_ABI void notify_one() _NOEXCEPT { std::__atomic_smart_ptr_notify_one(__fields_.__ctrl_address()); }
  _LIBCPP_HIDE_FROM_ABI void notify_all() _NOEXCEPT { std::__atomic_smart_ptr_notify_all(__fields_.__ctrl_address()); }

private:
  __atomic_smart_ptr_fields<_Tp> __fields_;
};

template <class _Tp>
struct atomic<weak_ptr<_Tp>> {
  using value_type = weak_ptr<_Tp>;

  static constexpr bool is_always_lock_free = false;

  _LIBCPP_HIDE_FROM_ABI atomic() _NOEXCEPT : __fields_(nullptr, nullptr) {}
  _LIBCPP_HIDE_FROM_ABI atomic(weak_ptr<_Tp> __desired) _NOEXCEPT : __fields_(__desired.__ptr_, __desired.__cntrl_) {
    __desired.__ptr_   = nullptr;
    __desired.__cntrl_ = nullptr;
  }

  atomic(const atomic&)            = delete;
  atomic& operator=(const atomic&) = delete;

  _LIBCPP_HIDE_FROM_ABI ~atomic() {
    if (auto* __c = __atomic_smart_ptr_storage::__decode(
            std::__cxx_atomic_load(__builtin_addressof(__fields_.__ctrl_), memory_order_relaxed)))
      __c->__release_weak();
  }

  _LIBCPP_HIDE_FROM_ABI bool is_lock_free() const _NOEXCEPT { return false; }

  _LIBCPP_HIDE_FROM_ABI void operator=(weak_ptr<_Tp> __desired) _NOEXCEPT { store(std::move(__desired)); }
  _LIBCPP_HIDE_FROM_ABI operator weak_ptr<_Tp>() const _NOEXCEPT { return load(); }

  _LIBCPP_HIDE_FROM_ABI void store(weak_ptr<_Tp> __desired, memory_order __m = memory_order_seq_cst) _NOEXCEPT
      _LIBCPP_CHECK_STORE_MEMORY_ORDER(__m) {
    (void)__m;
    _Tp* __desired_ptr               = __desired.__ptr_;
    __shared_weak_count* __desired_c = __desired.__cntrl_;
    __desired.__ptr_                 = nullptr;
    __desired.__cntrl_               = nullptr;

    __fields_.__lock();
    __shared_weak_count* __old_c = __atomic_smart_ptr_storage::__decode(
        std::__cxx_atomic_load(__builtin_addressof(__fields_.__ctrl_), memory_order_relaxed));
    std::__cxx_atomic_store(__builtin_addressof(__fields_.__ptr_), __desired_ptr, memory_order_relaxed);
    __fields_.__unlock(__desired_c);

    if (__old_c)
      __old_c->__release_weak();
  }

  _LIBCPP_HIDE_FROM_ABI weak_ptr<_Tp> load(memory_order __m = memory_order_seq_cst) const _NOEXCEPT
      _LIBCPP_CHECK_LOAD_MEMORY_ORDER(__m) {
    (void)__m;
    __fields_.__lock();
    _Tp* __ptr               = std::__cxx_atomic_load(__builtin_addressof(__fields_.__ptr_), memory_order_relaxed);
    __shared_weak_count* __c = __atomic_smart_ptr_storage::__decode(
        std::__cxx_atomic_load(__builtin_addressof(__fields_.__ctrl_), memory_order_relaxed));
    if (__c)
      __c->__add_weak();
    __fields_.__unlock(__c);
    return weak_ptr<_Tp>::__create_with_control_block(__ptr, __c);
  }

  _LIBCPP_HIDE_FROM_ABI weak_ptr<_Tp>
  exchange(weak_ptr<_Tp> __desired, memory_order __m = memory_order_seq_cst) _NOEXCEPT {
    (void)__m;
    _Tp* __desired_ptr               = __desired.__ptr_;
    __shared_weak_count* __desired_c = __desired.__cntrl_;
    __desired.__ptr_                 = nullptr;
    __desired.__cntrl_               = nullptr;

    __fields_.__lock();
    _Tp* __old_ptr               = std::__cxx_atomic_load(__builtin_addressof(__fields_.__ptr_), memory_order_relaxed);
    __shared_weak_count* __old_c = __atomic_smart_ptr_storage::__decode(
        std::__cxx_atomic_load(__builtin_addressof(__fields_.__ctrl_), memory_order_relaxed));
    std::__cxx_atomic_store(__builtin_addressof(__fields_.__ptr_), __desired_ptr, memory_order_relaxed);
    __fields_.__unlock(__desired_c);

    return weak_ptr<_Tp>::__create_with_control_block(__old_ptr, __old_c);
  }

  _LIBCPP_HIDE_FROM_ABI bool compare_exchange_strong(
      weak_ptr<_Tp>& __expected, weak_ptr<_Tp> __desired, memory_order __success, memory_order __failure) _NOEXCEPT
      _LIBCPP_CHECK_EXCHANGE_MEMORY_ORDER(__success, __failure) {
    (void)__success;
    (void)__failure;
    __fields_.__lock();
    _Tp* __cur_ptr               = std::__cxx_atomic_load(__builtin_addressof(__fields_.__ptr_), memory_order_relaxed);
    __shared_weak_count* __cur_c = __atomic_smart_ptr_storage::__decode(
        std::__cxx_atomic_load(__builtin_addressof(__fields_.__ctrl_), memory_order_relaxed));

    if (__atomic_smart_ptr_equivalent(__cur_ptr, __cur_c, __expected.__ptr_, __expected.__cntrl_)) {
      _Tp* __desired_ptr               = __desired.__ptr_;
      __shared_weak_count* __desired_c = __desired.__cntrl_;
      __desired.__ptr_                 = nullptr;
      __desired.__cntrl_               = nullptr;

      std::__cxx_atomic_store(__builtin_addressof(__fields_.__ptr_), __desired_ptr, memory_order_relaxed);
      __fields_.__unlock(__desired_c);
      if (__cur_c)
        __cur_c->__release_weak();
      return true;
    }

    if (__cur_c)
      __cur_c->__add_weak();
    __fields_.__unlock(__cur_c);
    __expected = weak_ptr<_Tp>::__create_with_control_block(__cur_ptr, __cur_c);
    return false;
  }

  _LIBCPP_HIDE_FROM_ABI bool compare_exchange_strong(
      weak_ptr<_Tp>& __expected, weak_ptr<_Tp> __desired, memory_order __m = memory_order_seq_cst) _NOEXCEPT {
    return compare_exchange_strong(__expected, std::move(__desired), __m, std::__to_failure_order(__m));
  }

  _LIBCPP_HIDE_FROM_ABI bool compare_exchange_weak(
      weak_ptr<_Tp>& __expected, weak_ptr<_Tp> __desired, memory_order __success, memory_order __failure) _NOEXCEPT
      _LIBCPP_CHECK_EXCHANGE_MEMORY_ORDER(__success, __failure) {
    return compare_exchange_strong(__expected, std::move(__desired), __success, __failure);
  }

  _LIBCPP_HIDE_FROM_ABI bool compare_exchange_weak(
      weak_ptr<_Tp>& __expected, weak_ptr<_Tp> __desired, memory_order __m = memory_order_seq_cst) _NOEXCEPT {
    return compare_exchange_strong(__expected, std::move(__desired), __m, std::__to_failure_order(__m));
  }

  _LIBCPP_HIDE_FROM_ABI void wait(weak_ptr<_Tp> __old, memory_order __m = memory_order_seq_cst) const _NOEXCEPT
      _LIBCPP_CHECK_WAIT_MEMORY_ORDER(__m) {
    _Tp* __old_ptr               = __old.__ptr_;
    __shared_weak_count* __old_c = __old.__cntrl_;

    std::__atomic_smart_ptr_wait_on_address(__fields_.__ctrl_address(), [&] {
      uintptr_t __word             = std::__cxx_atomic_load(__builtin_addressof(__fields_.__ctrl_), __m);
      __shared_weak_count* __cur_c = __atomic_smart_ptr_storage::__decode(__word);
      if (__cur_c != __old_c)
        return true;
      _Tp* __cur_ptr = std::__cxx_atomic_load(__builtin_addressof(__fields_.__ptr_), __m);
      return !__atomic_smart_ptr_equivalent(__cur_ptr, __cur_c, __old_ptr, __old_c);
    });
  }

  _LIBCPP_HIDE_FROM_ABI void notify_one() _NOEXCEPT { std::__atomic_smart_ptr_notify_one(__fields_.__ctrl_address()); }
  _LIBCPP_HIDE_FROM_ABI void notify_all() _NOEXCEPT { std::__atomic_smart_ptr_notify_all(__fields_.__ctrl_address()); }

private:
  __atomic_smart_ptr_fields<_Tp> __fields_;
};

#endif // _LIBCPP_STD_VER >= 20 && _LIBCPP_HAS_THREADS && _LIBCPP_HAS_ATOMIC_HEADER

_LIBCPP_END_NAMESPACE_STD

_LIBCPP_POP_MACROS

#endif // _LIBCPP___MEMORY_ATOMIC_SHARED_PTR_H
