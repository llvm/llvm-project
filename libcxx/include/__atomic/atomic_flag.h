//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___ATOMIC_ATOMIC_FLAG_H
#define _LIBCPP___ATOMIC_ATOMIC_FLAG_H

#include <__atomic/atomic_sync.h>
#include <__atomic/atomic_waitable_traits.h>
#include <__atomic/memory_order.h>
#include <__atomic/support.h>
#include <__config>
#include <__memory/addressof.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

struct atomic_flag {
  __cxx_atomic_impl<bool> __atomic_impl_;

  bool test(memory_order __m = memory_order_seq_cst) const volatile _NOEXCEPT {
    return __atomic_impl_.__atomic_load(__m);
  }
  bool test(memory_order __m = memory_order_seq_cst) const _NOEXCEPT { return __atomic_impl_.__atomic_load(__m); }

  bool test_and_set(memory_order __m = memory_order_seq_cst) volatile _NOEXCEPT {
    return __atomic_impl_.__atomic_exchange(true, __m);
  }
  bool test_and_set(memory_order __m = memory_order_seq_cst) _NOEXCEPT {
    return __atomic_impl_.__atomic_exchange(true, __m);
  }
  void clear(memory_order __m = memory_order_seq_cst) volatile _NOEXCEPT { __atomic_impl_.__atomic_store(false, __m); }
  void clear(memory_order __m = memory_order_seq_cst) _NOEXCEPT { __atomic_impl_.__atomic_store(false, __m); }

#if _LIBCPP_STD_VER >= 20
  void wait(bool __v, memory_order __m = memory_order_seq_cst) const volatile noexcept {
    std::__atomic_wait(*this, __v, __m);
  }
  void wait(bool __v, memory_order __m = memory_order_seq_cst) const noexcept { std::__atomic_wait(*this, __v, __m); }
  void notify_one() volatile noexcept { std::__atomic_notify_one(*this); }
  void notify_one() noexcept { std::__atomic_notify_one(*this); }
  void notify_all() volatile noexcept { std::__atomic_notify_all(*this); }
  void notify_all() noexcept { std::__atomic_notify_all(*this); }
#endif

#if _LIBCPP_STD_VER >= 20
  constexpr atomic_flag() noexcept : __atomic_impl_(false) {}
#else
  atomic_flag() _NOEXCEPT = default;
#endif

  _LIBCPP_CONSTEXPR atomic_flag(bool __b) _NOEXCEPT : __atomic_impl_(__b) {} // EXTENSION

  atomic_flag(const atomic_flag&)                     = delete;
  atomic_flag& operator=(const atomic_flag&)          = delete;
  atomic_flag& operator=(const atomic_flag&) volatile = delete;
};

#if _LIBCPP_STD_VER >= 20
template <>
struct __atomic_waitable_traits<atomic_flag> {
  using __value_type _LIBCPP_NODEBUG = bool;

  static bool __atomic_load(const atomic_flag& __a, memory_order __order) {
    return __a.__atomic_impl_.__atomic_load(__order);
  }

  static bool __atomic_load(const volatile atomic_flag& __a, memory_order __order) {
    return __a.__atomic_impl_.__atomic_load(__order);
  }

  static const __cxx_atomic_impl<bool>* __atomic_contention_address(const atomic_flag& __a) {
    return std::addressof(__a.__atomic_impl_);
  }

  static const volatile __cxx_atomic_impl<bool>* __atomic_contention_address(const volatile atomic_flag& __a) {
    return std::addressof(__a.__atomic_impl_);
  }
};
#endif // _LIBCPP_STD_VER >= 20

inline bool atomic_flag_test(const volatile atomic_flag* __o) _NOEXCEPT { return __o->test(); }

inline bool atomic_flag_test(const atomic_flag* __o) _NOEXCEPT { return __o->test(); }

inline bool atomic_flag_test_explicit(const volatile atomic_flag* __o, memory_order __m) _NOEXCEPT {
  return __o->test(__m);
}

inline bool atomic_flag_test_explicit(const atomic_flag* __o, memory_order __m) _NOEXCEPT { return __o->test(__m); }

inline bool atomic_flag_test_and_set(volatile atomic_flag* __o) _NOEXCEPT { return __o->test_and_set(); }

inline bool atomic_flag_test_and_set(atomic_flag* __o) _NOEXCEPT { return __o->test_and_set(); }

inline bool atomic_flag_test_and_set_explicit(volatile atomic_flag* __o, memory_order __m) _NOEXCEPT {
  return __o->test_and_set(__m);
}

inline bool atomic_flag_test_and_set_explicit(atomic_flag* __o, memory_order __m) _NOEXCEPT {
  return __o->test_and_set(__m);
}

inline void atomic_flag_clear(volatile atomic_flag* __o) _NOEXCEPT { __o->clear(); }

inline void atomic_flag_clear(atomic_flag* __o) _NOEXCEPT { __o->clear(); }

inline void atomic_flag_clear_explicit(volatile atomic_flag* __o, memory_order __m) _NOEXCEPT { __o->clear(__m); }

inline void atomic_flag_clear_explicit(atomic_flag* __o, memory_order __m) _NOEXCEPT { __o->clear(__m); }

#if _LIBCPP_STD_VER >= 20
inline void atomic_flag_wait(const volatile atomic_flag* __o, bool __v) noexcept { __o->wait(__v); }

inline void atomic_flag_wait(const atomic_flag* __o, bool __v) noexcept { __o->wait(__v); }

inline void atomic_flag_wait_explicit(const volatile atomic_flag* __o, bool __v, memory_order __m) noexcept {
  __o->wait(__v, __m);
}

inline void atomic_flag_wait_explicit(const atomic_flag* __o, bool __v, memory_order __m) noexcept {
  __o->wait(__v, __m);
}

inline void atomic_flag_notify_one(volatile atomic_flag* __o) noexcept { __o->notify_one(); }
inline void atomic_flag_notify_one(atomic_flag* __o) noexcept { __o->notify_one(); }
inline void atomic_flag_notify_all(volatile atomic_flag* __o) noexcept { __o->notify_all(); }
inline void atomic_flag_notify_all(atomic_flag* __o) noexcept { __o->notify_all(); }
#endif // _LIBCPP_STD_VER >= 20

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___ATOMIC_ATOMIC_FLAG_H
