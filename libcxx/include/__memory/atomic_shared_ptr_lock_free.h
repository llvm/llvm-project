// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___MEMORY_ATOMIC_SHARED_PTR_LOCK_FREE_H
#define _LIBCPP___MEMORY_ATOMIC_SHARED_PTR_LOCK_FREE_H

#include <__atomic/atomic_sync_lite.h>
#include <__atomic/check_memory_order.h>
#include <__atomic/memory_order.h>
#include <__atomic/support.h>
#include <__atomic/to_failure_order.h>
#include <__atomic/to_gcc_order.h>
#include <__config>
#include <__cstddef/nullptr_t.h>
#include <__memory/shared_count.h>
#include <__utility/move.h>

#include <cstdint>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_PUSH_MACROS
#include <__undef_macros>

_LIBCPP_BEGIN_NAMESPACE_STD

#if _LIBCPP_STD_VER >= 20 && _LIBCPP_HAS_THREADS && _LIBCPP_HAS_ATOMIC_HEADER

template <class _Tp>
class shared_ptr;

template <class _Tp>
class weak_ptr;

template <class _Tp>
struct atomic;

// __ctrl_ word layout (8 bytes):
//   bits 0..1    always zero (CBs are at least 4-byte aligned)
//   bits 2..47   control-block pointer
//   bits 48..63  16-bit local refcount (Williams split-refcount aspiration)
// Bits 48..63 are available on both DWCAS-capable ISAs (x86-64 without LAMA,
// AArch64 without LVA): user-space pointers stay within 48 bits.
struct __atomic_smart_ptr_storage {
  static constexpr uintptr_t __lock_bit_        = uintptr_t{1};
  static constexpr uintptr_t __notify_bit_      = uintptr_t{2};
  static constexpr uintptr_t __ptr_mask_        = ~(__lock_bit_ | __notify_bit_);
  static constexpr unsigned __local_count_shift = 48;
  static constexpr uintptr_t __local_count_unit = uintptr_t{1} << __local_count_shift;
  static constexpr uintptr_t __local_count_mask = uintptr_t{0xFFFF} << __local_count_shift;
  static constexpr uintptr_t __dwcas_ptr_mask   = __ptr_mask_ & ~__local_count_mask;
  static constexpr uint16_t __local_count_max   = 0xFFFF;

  static uintptr_t __encode_dwcas(__shared_weak_count* __ctrl, uint16_t __local) noexcept {
    return (reinterpret_cast<uintptr_t>(__ctrl) & __dwcas_ptr_mask) |
           (static_cast<uintptr_t>(__local) << __local_count_shift);
  }

  static __shared_weak_count* __decode_dwcas(uintptr_t __word) noexcept {
    return reinterpret_cast<__shared_weak_count*>(__word & __dwcas_ptr_mask);
  }

  static uint16_t __decode_local(uintptr_t __word) noexcept {
    return static_cast<uint16_t>((__word & __local_count_mask) >> __local_count_shift);
  }
};

inline void __atomic_smart_ptr_notify_one(const void* __address) noexcept {
#  if _LIBCPP_AVAILABILITY_HAS_NEW_SYNC
  std::__atomic_notify_one_global_table(__address);
#  else
  std::__cxx_atomic_notify_one(reinterpret_cast<void const volatile*>(__address));
#  endif
}

inline void __atomic_smart_ptr_notify_all(const void* __address) noexcept {
#  if _LIBCPP_AVAILABILITY_HAS_NEW_SYNC
  std::__atomic_notify_all_global_table(__address);
#  else
  std::__cxx_atomic_notify_all(reinterpret_cast<void const volatile*>(__address));
#  endif
}

template <class _Poll>
inline void __atomic_smart_ptr_wait_on_address(const void* __address, const _Poll& __poll) noexcept {
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

// CMPXCHG16B/CASP require 16-byte alignment; the public ABI size is 16 bytes.
template <class _Element>
struct alignas(16) __atomic_smart_ptr_fields {
  mutable __cxx_atomic_impl<_Element*> __ptr_;
  mutable __cxx_atomic_impl<uintptr_t> __ctrl_;

  __atomic_smart_ptr_fields(_Element* __p, __shared_weak_count* __c) noexcept
      : __ptr_(__p), __ctrl_(__atomic_smart_ptr_storage::__encode_dwcas(__c, 0)) {}

  // DWCAS updates __ptr_ and __ctrl_ as one 16-byte pair, so waiters observe
  // the struct base; watching only __ctrl_ would miss stores where the pointer
  // changes but the CB stays the same (e.g., aliasing constructor).
  const void* __wait_address() const noexcept { return static_cast<const void*>(this); }

  __extension__ using __sp_u128 = unsigned __int128;

  // Storage layout: low 64 bits = __ptr_, high 64 bits = __ctrl_. Going
  // through `this` instead of __builtin_addressof(__ptr_) keeps the 16-byte alignment explicit
  // and avoids implementation-defined aliasing between the sub-atomics and
  // the wide view (sub-atomics are never accessed when DWCAS is enabled).
  // TODO(review): this cast bypasses strict aliasing - valid only because
  // __ptr_ and __ctrl_ are never touched via __cxx_atomic_* on this path.
  __sp_u128* __dwcas_address() const noexcept {
    return const_cast<__sp_u128*>(reinterpret_cast<const __sp_u128*>(this));
  }

  __sp_u128 __dwcas_load(memory_order __o) const noexcept {
    __sp_u128 __word;
    __atomic_load(__dwcas_address(), __builtin_addressof(__word), std::__to_gcc_order(__o));
    return __word;
  }

  void __dwcas_store(__sp_u128 __word, memory_order __o) const noexcept {
    __atomic_store(__dwcas_address(), __builtin_addressof(__word), std::__to_gcc_order(__o));
  }

  __sp_u128 __dwcas_exchange(__sp_u128 __new, memory_order __o) const noexcept {
    __sp_u128 __old;
    __atomic_exchange(
        __dwcas_address(), __builtin_addressof(__new), __builtin_addressof(__old), std::__to_gcc_order(__o));
    return __old;
  }

  // Weak CAS suffices for both strong and weak public variants - the caller
  // controls retry policy (see __cas_dwcas below).
  bool __dwcas_compare_exchange_weak(
      __sp_u128& __expected, __sp_u128 __desired, memory_order __s, memory_order __f) const noexcept {
    return __atomic_compare_exchange(
        __dwcas_address(),
        __builtin_addressof(__expected),
        __builtin_addressof(__desired),
        /*weak=*/true,
        std::__to_gcc_order(__s),
        std::__to_gcc_failure_order(__f));
  }

  static _Element* __pair_ptr(__sp_u128 __w) noexcept {
    return reinterpret_cast<_Element*>(static_cast<uintptr_t>(__w));
  }
  static uintptr_t __pair_ctrl(__sp_u128 __w) noexcept { return static_cast<uintptr_t>(__w >> 64); }
  static __sp_u128 __pair_make(_Element* __p, uintptr_t __c) noexcept {
    return static_cast<__sp_u128>(reinterpret_cast<uintptr_t>(__p)) | (static_cast<__sp_u128>(__c) << 64);
  }
};

// [util.smartptr.atomic.shared]: same stored pointer and same ownership, or both empty.
template <class _Element>
inline bool __atomic_smart_ptr_equivalent(
    _Element* __ptr,
    __shared_weak_count* __ctrl,
    _Element* __expected_ptr,
    __shared_weak_count* __expected_ctrl) noexcept {
  if (__ctrl == nullptr && __expected_ctrl == nullptr)
    return true;
  return __ptr == __expected_ptr && __ctrl == __expected_ctrl;
}

// Pre-pay the per-in-flight-load fetch_add(1) tick that each aspirational
// loader still owes the retiring control block; without this, the racing
// loader would double-count its own contribution on the new block.
template <class _Counter>
inline void __atomic_smart_ptr_drain_shared(_Counter* __cb, uint16_t __local) noexcept {
  for (uint16_t __i = 0; __i < __local; ++__i)
    __cb->__add_shared();
}

template <class _Counter>
inline void __atomic_smart_ptr_drain_weak(_Counter* __cb, uint16_t __local) noexcept {
  for (uint16_t __i = 0; __i < __local; ++__i)
    __cb->__add_weak();
}

template <class _Tp>
struct atomic<shared_ptr<_Tp>> {
  using value_type = shared_ptr<_Tp>;

  // is_always_lock_free stays false - the umbrella header can dispatch to
  // the lock-based implementation on cx16/LSE-absent builds; within this TU
  // is_lock_free() is always true.
  static constexpr bool is_always_lock_free = false;

  atomic() noexcept : __fields_(nullptr, nullptr) {}
  constexpr atomic(nullptr_t) noexcept : __fields_(nullptr, nullptr) {}
  atomic(shared_ptr<_Tp> __desired) noexcept : __fields_(__desired.__ptr_, __desired.__cntrl_) {
    __desired.__ptr_   = nullptr;
    __desired.__cntrl_ = nullptr;
  }

  atomic(const atomic&)            = delete;
  atomic& operator=(const atomic&) = delete;

  ~atomic() {
    using __fields = __atomic_smart_ptr_fields<_Tp>;
    auto __word    = __fields_.__dwcas_load(memory_order_relaxed);
    if (auto* __c = __atomic_smart_ptr_storage::__decode_dwcas(__fields::__pair_ctrl(__word))) {
      // No in-flight loads can exist at destruction time (the standard
      // forbids concurrent operations on a destroyed atomic), so local
      // count must be zero by construction.
      __c->__release_shared();
    }
  }

  [[nodiscard]] bool is_lock_free() const noexcept { return true; }

  void operator=(shared_ptr<_Tp> __desired) noexcept { store(std::move(__desired)); }
  void operator=(nullptr_t) noexcept { store(nullptr); }
  operator shared_ptr<_Tp>() const noexcept { return load(); }

  void store(shared_ptr<_Tp> __desired, memory_order __m = memory_order_seq_cst) noexcept
      _LIBCPP_CHECK_STORE_MEMORY_ORDER(__m) {
    __store_dwcas(std::move(__desired), __m);
  }

  [[nodiscard]] shared_ptr<_Tp> load(memory_order __m = memory_order_seq_cst) const noexcept
      _LIBCPP_CHECK_LOAD_MEMORY_ORDER(__m) {
    return __load_dwcas(__m);
  }

  shared_ptr<_Tp> exchange(shared_ptr<_Tp> __desired, memory_order __m = memory_order_seq_cst) noexcept {
    return __exchange_dwcas(std::move(__desired), __m);
  }

  bool compare_exchange_strong(
      shared_ptr<_Tp>& __expected, shared_ptr<_Tp> __desired, memory_order __success, memory_order __failure) noexcept
      _LIBCPP_CHECK_EXCHANGE_MEMORY_ORDER(__success, __failure) {
    return __cas_dwcas(__expected, std::move(__desired), __success, __failure, /*__is_strong=*/true);
  }

  bool compare_exchange_strong(
      shared_ptr<_Tp>& __expected, shared_ptr<_Tp> __desired, memory_order __m = memory_order_seq_cst) noexcept {
    return compare_exchange_strong(__expected, std::move(__desired), __m, std::__to_failure_order(__m));
  }

  bool compare_exchange_weak(
      shared_ptr<_Tp>& __expected, shared_ptr<_Tp> __desired, memory_order __success, memory_order __failure) noexcept
      _LIBCPP_CHECK_EXCHANGE_MEMORY_ORDER(__success, __failure) {
    return __cas_dwcas(__expected, std::move(__desired), __success, __failure, /*__is_strong=*/false);
  }

  bool compare_exchange_weak(
      shared_ptr<_Tp>& __expected, shared_ptr<_Tp> __desired, memory_order __m = memory_order_seq_cst) noexcept {
    return compare_exchange_weak(__expected, std::move(__desired), __m, std::__to_failure_order(__m));
  }

  void wait(shared_ptr<_Tp> __old, memory_order __m = memory_order_seq_cst) const noexcept
      _LIBCPP_CHECK_WAIT_MEMORY_ORDER(__m) {
    _Tp* __old_ptr               = __old.__ptr_;
    __shared_weak_count* __old_c = __old.__cntrl_;

    std::__atomic_smart_ptr_wait_on_address(__fields_.__wait_address(), [&] {
      using __fields               = __atomic_smart_ptr_fields<_Tp>;
      auto __word                  = __fields_.__dwcas_load(__m);
      __shared_weak_count* __cur_c = __atomic_smart_ptr_storage::__decode_dwcas(__fields::__pair_ctrl(__word));
      _Tp* __cur_ptr               = __fields::__pair_ptr(__word);
      return !__atomic_smart_ptr_equivalent(__cur_ptr, __cur_c, __old_ptr, __old_c);
    });
  }

  void notify_one() noexcept { std::__atomic_smart_ptr_notify_one(__fields_.__wait_address()); }
  void notify_all() noexcept { std::__atomic_smart_ptr_notify_all(__fields_.__wait_address()); }

private:
  __atomic_smart_ptr_fields<_Tp> __fields_;

  using __fields_type = __atomic_smart_ptr_fields<_Tp>;
  using __u128        = typename __fields_type::__sp_u128;

  shared_ptr<_Tp> __load_dwcas(memory_order __m) const noexcept {
    auto __expected = __fields_.__dwcas_load(memory_order_acquire);

    __shared_weak_count* __claimed_cb;
    _Tp* __claimed_ptr;

    while (true) {
      uintptr_t __ctrl_word = __fields_type::__pair_ctrl(__expected);
      __claimed_cb          = __atomic_smart_ptr_storage::__decode_dwcas(__ctrl_word);
      __claimed_ptr         = __fields_type::__pair_ptr(__expected);

      if (__claimed_cb == nullptr)
        return shared_ptr<_Tp>{};

      uint16_t __local = __atomic_smart_ptr_storage::__decode_local(__ctrl_word);
      if (__local == __atomic_smart_ptr_storage::__local_count_max) {
        // Local count is saturated; yielding here is bounded because progress
        // on another thread will lower it back. Wider counters would only push
        // the threshold further out - they cannot eliminate the retry.
        __expected = __fields_.__dwcas_load(memory_order_acquire);
        continue;
      }

      // TODO(question): ABA: the 16-bit local count is not a version tag across CB
      // lifecycles. A freed CB reallocated at the same address with the same count
      // lets this CAS succeed spuriously. Is this risk acceptable?
      auto __desired =
          __fields_type::__pair_make(__claimed_ptr, __ctrl_word + __atomic_smart_ptr_storage::__local_count_unit);
      if (__fields_.__dwcas_compare_exchange_weak(__expected, __desired, memory_order_acq_rel, memory_order_acquire))
        break;
    }

    __claimed_cb->__add_shared();

    auto __cur = __fields_.__dwcas_load(memory_order_acquire);
    while (true) {
      uintptr_t __cur_ctrl          = __fields_type::__pair_ctrl(__cur);
      __shared_weak_count* __cur_cb = __atomic_smart_ptr_storage::__decode_dwcas(__cur_ctrl);

      // TODO(review): the store that replaced this CB pre-paid our tick via
      // drain_shared, so __release_shared below cannot be the last ref - verify
      // no concurrent release can create a zero-count window between drain and here.
      if (__cur_cb != __claimed_cb) {
        // A racing store already drained our local tick via __drain_shared,
        // so __add_shared above became a double-count. Cancel it.
        __claimed_cb->__release_shared();
        break;
      }

      auto __dec = __fields_type::__pair_make(
          __fields_type::__pair_ptr(__cur), __cur_ctrl - __atomic_smart_ptr_storage::__local_count_unit);
      if (__fields_.__dwcas_compare_exchange_weak(__cur, __dec, memory_order_acq_rel, memory_order_acquire))
        break;
    }

    (void)__m;
    return shared_ptr<_Tp>::__create_with_control_block(__claimed_ptr, __claimed_cb);
  }

  void __store_dwcas(shared_ptr<_Tp> __desired, memory_order __m) noexcept {
    _Tp* __desired_ptr               = __desired.__ptr_;
    __shared_weak_count* __desired_c = __desired.__cntrl_;
    __desired.__ptr_                 = nullptr;
    __desired.__cntrl_               = nullptr;

    auto __new_pair =
        __fields_type::__pair_make(__desired_ptr, __atomic_smart_ptr_storage::__encode_dwcas(__desired_c, 0));
    auto __old_pair = __fields_.__dwcas_exchange(__new_pair, memory_order_acq_rel);

    __retire_old_pair(__old_pair);

    (void)__m;
    std::__atomic_smart_ptr_notify_all(__fields_.__wait_address());
  }

  shared_ptr<_Tp> __exchange_dwcas(shared_ptr<_Tp> __desired, memory_order __m) noexcept {
    _Tp* __desired_ptr               = __desired.__ptr_;
    __shared_weak_count* __desired_c = __desired.__cntrl_;
    __desired.__ptr_                 = nullptr;
    __desired.__cntrl_               = nullptr;

    auto __new_pair =
        __fields_type::__pair_make(__desired_ptr, __atomic_smart_ptr_storage::__encode_dwcas(__desired_c, 0));
    auto __old_pair = __fields_.__dwcas_exchange(__new_pair, memory_order_acq_rel);

    uintptr_t __old_ctrl          = __fields_type::__pair_ctrl(__old_pair);
    _Tp* __old_ptr                = __fields_type::__pair_ptr(__old_pair);
    __shared_weak_count* __old_cb = __atomic_smart_ptr_storage::__decode_dwcas(__old_ctrl);
    uint16_t __old_local          = __atomic_smart_ptr_storage::__decode_local(__old_ctrl);

    if (__old_cb) {
      // Drain in-flight loaders' aspirational ticks. After this, __old_cb's
      // global count includes one ref per in-flight load that paired up with
      // this CB. We then transfer ownership of the atomic's own ref to the
      // returned shared_ptr (no add/release).
      std::__atomic_smart_ptr_drain_shared(__old_cb, __old_local);
    }

    (void)__m;
    std::__atomic_smart_ptr_notify_all(__fields_.__wait_address());

    return shared_ptr<_Tp>::__create_with_control_block(__old_ptr, __old_cb);
  }

  // __is_strong: when true, retry on benign DWCAS failures (e.g., local-count
  //           changes by concurrent loads); only return false on a real
  //           ownership-equivalence mismatch.
  bool __cas_dwcas(shared_ptr<_Tp>& __expected,
                   shared_ptr<_Tp> __desired,
                   memory_order __success,
                   memory_order __failure,
                   bool __is_strong) noexcept {
    _Tp* __exp_ptr               = __expected.__ptr_;
    __shared_weak_count* __exp_c = __expected.__cntrl_;

    _Tp* __des_ptr               = __desired.__ptr_;
    __shared_weak_count* __des_c = __desired.__cntrl_;

    auto __cur = __fields_.__dwcas_load(memory_order_acquire);

    while (true) {
      uintptr_t __cur_ctrl          = __fields_type::__pair_ctrl(__cur);
      _Tp* __cur_ptr                = __fields_type::__pair_ptr(__cur);
      __shared_weak_count* __cur_cb = __atomic_smart_ptr_storage::__decode_dwcas(__cur_ctrl);

      if (!__atomic_smart_ptr_equivalent(__cur_ptr, __cur_cb, __exp_ptr, __exp_c)) {
        // Ownership mismatch. Update __expected to the actual current value
        // (with a fresh shared_ptr reference). __desired is NOT consumed;
        // its destructor will release its own ref.
        (void)__success;
        __expected = __load_dwcas(__failure);
        return false;
      }

      // Equivalence holds. Try to publish __desired with local_count=0.
      auto __desired_pair =
          __fields_type::__pair_make(__des_ptr, __atomic_smart_ptr_storage::__encode_dwcas(__des_c, 0));

      if (__fields_.__dwcas_compare_exchange_weak(__cur, __desired_pair, __success, __failure)) {
        uint16_t __cur_local = __atomic_smart_ptr_storage::__decode_local(__cur_ctrl);
        if (__cur_cb) {
          std::__atomic_smart_ptr_drain_shared(__cur_cb, __cur_local);
          __cur_cb->__release_shared();
        }
        __desired.__ptr_   = nullptr;
        __desired.__cntrl_ = nullptr;
        std::__atomic_smart_ptr_notify_all(__fields_.__wait_address());
        return true;
      }

      if (!__is_strong) {
        // weak: a single underlying CAS attempt - if it failed for any
        // reason (real ownership change or concurrent local-count update),
        // report failure with the current pair as the new __expected.
        __expected = __load_dwcas(__failure);
        return false;
      }
      // Strong CAS: loop and re-check equivalence.
    }
  }

  // Retire path used by __store_dwcas - drains in-flight ticks and releases
  // the atomic's own ref on the old control block.
  void __retire_old_pair(__u128 __old_pair) noexcept {
    uintptr_t __old_ctrl          = __fields_type::__pair_ctrl(__old_pair);
    __shared_weak_count* __old_cb = __atomic_smart_ptr_storage::__decode_dwcas(__old_ctrl);
    uint16_t __old_local          = __atomic_smart_ptr_storage::__decode_local(__old_ctrl);

    if (__old_cb == nullptr)
      return;

    // Pre-pay aspirational ticks of in-flight loaders.
    std::__atomic_smart_ptr_drain_shared(__old_cb, __old_local);
    // Release the ref this atomic held on the old control block.
    __old_cb->__release_shared();
  }
};

template <class _Tp>
struct atomic<weak_ptr<_Tp>> {
  using value_type = weak_ptr<_Tp>;

  static constexpr bool is_always_lock_free = false;

  atomic() noexcept : __fields_(nullptr, nullptr) {}
  atomic(weak_ptr<_Tp> __desired) noexcept : __fields_(__desired.__ptr_, __desired.__cntrl_) {
    __desired.__ptr_   = nullptr;
    __desired.__cntrl_ = nullptr;
  }

  atomic(const atomic&)            = delete;
  atomic& operator=(const atomic&) = delete;

  ~atomic() {
    using __fields = __atomic_smart_ptr_fields<_Tp>;
    auto __word    = __fields_.__dwcas_load(memory_order_relaxed);
    if (auto* __c = __atomic_smart_ptr_storage::__decode_dwcas(__fields::__pair_ctrl(__word)))
      __c->__release_weak();
  }

  [[nodiscard]] bool is_lock_free() const noexcept { return true; }

  void operator=(weak_ptr<_Tp> __desired) noexcept { store(std::move(__desired)); }
  operator weak_ptr<_Tp>() const noexcept { return load(); }

  void store(weak_ptr<_Tp> __desired, memory_order __m = memory_order_seq_cst) noexcept
      _LIBCPP_CHECK_STORE_MEMORY_ORDER(__m) {
    __store_dwcas(std::move(__desired), __m);
  }

  [[nodiscard]] weak_ptr<_Tp> load(memory_order __m = memory_order_seq_cst) const noexcept
      _LIBCPP_CHECK_LOAD_MEMORY_ORDER(__m) {
    return __load_dwcas(__m);
  }

  weak_ptr<_Tp> exchange(weak_ptr<_Tp> __desired, memory_order __m = memory_order_seq_cst) noexcept {
    return __exchange_dwcas(std::move(__desired), __m);
  }

  bool compare_exchange_strong(
      weak_ptr<_Tp>& __expected, weak_ptr<_Tp> __desired, memory_order __success, memory_order __failure) noexcept
      _LIBCPP_CHECK_EXCHANGE_MEMORY_ORDER(__success, __failure) {
    return __cas_dwcas(__expected, std::move(__desired), __success, __failure, /*__is_strong=*/true);
  }

  bool compare_exchange_strong(
      weak_ptr<_Tp>& __expected, weak_ptr<_Tp> __desired, memory_order __m = memory_order_seq_cst) noexcept {
    return compare_exchange_strong(__expected, std::move(__desired), __m, std::__to_failure_order(__m));
  }

  bool compare_exchange_weak(
      weak_ptr<_Tp>& __expected, weak_ptr<_Tp> __desired, memory_order __success, memory_order __failure) noexcept
      _LIBCPP_CHECK_EXCHANGE_MEMORY_ORDER(__success, __failure) {
    return __cas_dwcas(__expected, std::move(__desired), __success, __failure, /*__is_strong=*/false);
  }

  bool compare_exchange_weak(
      weak_ptr<_Tp>& __expected, weak_ptr<_Tp> __desired, memory_order __m = memory_order_seq_cst) noexcept {
    return compare_exchange_weak(__expected, std::move(__desired), __m, std::__to_failure_order(__m));
  }

  void wait(weak_ptr<_Tp> __old, memory_order __m = memory_order_seq_cst) const noexcept
      _LIBCPP_CHECK_WAIT_MEMORY_ORDER(__m) {
    _Tp* __old_ptr               = __old.__ptr_;
    __shared_weak_count* __old_c = __old.__cntrl_;

    std::__atomic_smart_ptr_wait_on_address(__fields_.__wait_address(), [&] {
      using __fields               = __atomic_smart_ptr_fields<_Tp>;
      auto __word                  = __fields_.__dwcas_load(__m);
      __shared_weak_count* __cur_c = __atomic_smart_ptr_storage::__decode_dwcas(__fields::__pair_ctrl(__word));
      _Tp* __cur_ptr               = __fields::__pair_ptr(__word);
      return !__atomic_smart_ptr_equivalent(__cur_ptr, __cur_c, __old_ptr, __old_c);
    });
  }

  void notify_one() noexcept { std::__atomic_smart_ptr_notify_one(__fields_.__wait_address()); }
  void notify_all() noexcept { std::__atomic_smart_ptr_notify_all(__fields_.__wait_address()); }

private:
  __atomic_smart_ptr_fields<_Tp> __fields_;

  using __fields_type = __atomic_smart_ptr_fields<_Tp>;
  using __u128        = typename __fields_type::__sp_u128;

  weak_ptr<_Tp> __load_dwcas(memory_order __m) const noexcept {
    auto __expected = __fields_.__dwcas_load(memory_order_acquire);

    __shared_weak_count* __claimed_cb;
    _Tp* __claimed_ptr;

    while (true) {
      uintptr_t __ctrl_word = __fields_type::__pair_ctrl(__expected);
      __claimed_cb          = __atomic_smart_ptr_storage::__decode_dwcas(__ctrl_word);
      __claimed_ptr         = __fields_type::__pair_ptr(__expected);

      if (__claimed_cb == nullptr)
        return weak_ptr<_Tp>{};

      uint16_t __local = __atomic_smart_ptr_storage::__decode_local(__ctrl_word);
      if (__local == __atomic_smart_ptr_storage::__local_count_max) {
        __expected = __fields_.__dwcas_load(memory_order_acquire);
        continue;
      }

      auto __desired =
          __fields_type::__pair_make(__claimed_ptr, __ctrl_word + __atomic_smart_ptr_storage::__local_count_unit);
      if (__fields_.__dwcas_compare_exchange_weak(__expected, __desired, memory_order_acq_rel, memory_order_acquire))
        break;
    }

    __claimed_cb->__add_weak();

    auto __cur = __fields_.__dwcas_load(memory_order_acquire);
    while (true) {
      uintptr_t __cur_ctrl          = __fields_type::__pair_ctrl(__cur);
      __shared_weak_count* __cur_cb = __atomic_smart_ptr_storage::__decode_dwcas(__cur_ctrl);

      if (__cur_cb != __claimed_cb) {
        __claimed_cb->__release_weak();
        break;
      }

      auto __dec = __fields_type::__pair_make(
          __fields_type::__pair_ptr(__cur), __cur_ctrl - __atomic_smart_ptr_storage::__local_count_unit);
      if (__fields_.__dwcas_compare_exchange_weak(__cur, __dec, memory_order_acq_rel, memory_order_acquire))
        break;
    }

    (void)__m;
    return weak_ptr<_Tp>::__create_with_control_block(__claimed_ptr, __claimed_cb);
  }

  void __store_dwcas(weak_ptr<_Tp> __desired, memory_order __m) noexcept {
    _Tp* __desired_ptr               = __desired.__ptr_;
    __shared_weak_count* __desired_c = __desired.__cntrl_;
    __desired.__ptr_                 = nullptr;
    __desired.__cntrl_               = nullptr;

    auto __new_pair =
        __fields_type::__pair_make(__desired_ptr, __atomic_smart_ptr_storage::__encode_dwcas(__desired_c, 0));
    auto __old_pair = __fields_.__dwcas_exchange(__new_pair, memory_order_acq_rel);

    __retire_old_pair_weak(__old_pair);

    (void)__m;
    std::__atomic_smart_ptr_notify_all(__fields_.__wait_address());
  }

  weak_ptr<_Tp> __exchange_dwcas(weak_ptr<_Tp> __desired, memory_order __m) noexcept {
    _Tp* __desired_ptr               = __desired.__ptr_;
    __shared_weak_count* __desired_c = __desired.__cntrl_;
    __desired.__ptr_                 = nullptr;
    __desired.__cntrl_               = nullptr;

    auto __new_pair =
        __fields_type::__pair_make(__desired_ptr, __atomic_smart_ptr_storage::__encode_dwcas(__desired_c, 0));
    auto __old_pair = __fields_.__dwcas_exchange(__new_pair, memory_order_acq_rel);

    uintptr_t __old_ctrl          = __fields_type::__pair_ctrl(__old_pair);
    _Tp* __old_ptr                = __fields_type::__pair_ptr(__old_pair);
    __shared_weak_count* __old_cb = __atomic_smart_ptr_storage::__decode_dwcas(__old_ctrl);
    uint16_t __old_local          = __atomic_smart_ptr_storage::__decode_local(__old_ctrl);

    if (__old_cb)
      std::__atomic_smart_ptr_drain_weak(__old_cb, __old_local);

    (void)__m;
    std::__atomic_smart_ptr_notify_all(__fields_.__wait_address());

    return weak_ptr<_Tp>::__create_with_control_block(__old_ptr, __old_cb);
  }

  bool __cas_dwcas(weak_ptr<_Tp>& __expected,
                   weak_ptr<_Tp> __desired,
                   memory_order __success,
                   memory_order __failure,
                   bool __is_strong) noexcept {
    _Tp* __exp_ptr               = __expected.__ptr_;
    __shared_weak_count* __exp_c = __expected.__cntrl_;

    _Tp* __des_ptr               = __desired.__ptr_;
    __shared_weak_count* __des_c = __desired.__cntrl_;

    auto __cur = __fields_.__dwcas_load(memory_order_acquire);

    while (true) {
      uintptr_t __cur_ctrl          = __fields_type::__pair_ctrl(__cur);
      _Tp* __cur_ptr                = __fields_type::__pair_ptr(__cur);
      __shared_weak_count* __cur_cb = __atomic_smart_ptr_storage::__decode_dwcas(__cur_ctrl);

      if (!__atomic_smart_ptr_equivalent(__cur_ptr, __cur_cb, __exp_ptr, __exp_c)) {
        (void)__success;
        __expected = __load_dwcas(__failure);
        return false;
      }

      auto __desired_pair =
          __fields_type::__pair_make(__des_ptr, __atomic_smart_ptr_storage::__encode_dwcas(__des_c, 0));

      if (__fields_.__dwcas_compare_exchange_weak(__cur, __desired_pair, __success, __failure)) {
        uint16_t __cur_local = __atomic_smart_ptr_storage::__decode_local(__cur_ctrl);
        if (__cur_cb) {
          std::__atomic_smart_ptr_drain_weak(__cur_cb, __cur_local);
          __cur_cb->__release_weak();
        }
        __desired.__ptr_   = nullptr;
        __desired.__cntrl_ = nullptr;
        std::__atomic_smart_ptr_notify_all(__fields_.__wait_address());
        return true;
      }

      if (!__is_strong) {
        __expected = __load_dwcas(__failure);
        return false;
      }
    }
  }

  void __retire_old_pair_weak(__u128 __old_pair) noexcept {
    uintptr_t __old_ctrl          = __fields_type::__pair_ctrl(__old_pair);
    __shared_weak_count* __old_cb = __atomic_smart_ptr_storage::__decode_dwcas(__old_ctrl);
    uint16_t __old_local          = __atomic_smart_ptr_storage::__decode_local(__old_ctrl);

    if (__old_cb == nullptr)
      return;

    std::__atomic_smart_ptr_drain_weak(__old_cb, __old_local);
    __old_cb->__release_weak();
  }
};

#endif // _LIBCPP_STD_VER >= 20 && _LIBCPP_HAS_THREADS && _LIBCPP_HAS_ATOMIC_HEADER

_LIBCPP_END_NAMESPACE_STD

_LIBCPP_POP_MACROS

#endif // _LIBCPP___MEMORY_ATOMIC_SHARED_PTR_LOCK_FREE_H
