//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___ATOMIC_SUPPORT_GCC_H
#define _LIBCPP___ATOMIC_SUPPORT_GCC_H

#include <__atomic/clear_padding.h>
#include <__atomic/memory_order.h>
#include <__atomic/to_gcc_order.h>
#include <__config>
#include <__memory/addressof.h>
#include <__type_traits/enable_if.h>
#include <__type_traits/is_assignable.h>
#include <__type_traits/remove_const.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

//
// This file implements support for GCC-style atomics
//

_LIBCPP_BEGIN_NAMESPACE_STD

template <typename _Tp>
struct __skip_amt {
  enum { value = 1 };
};

template <typename _Tp>
struct __skip_amt<_Tp*> {
  enum { value = sizeof(_Tp) };
};

// FIXME: Haven't figured out what the spec says about using arrays with
// atomic_fetch_add. Force a failure rather than creating bad behavior.
template <typename _Tp>
struct __skip_amt<_Tp[]> {};
template <typename _Tp, int _Np>
struct __skip_amt<_Tp[_Np]> {};

template <typename _Tp>
struct __cxx_atomic_base_impl {
#ifndef _LIBCPP_CXX03_LANG
  __cxx_atomic_base_impl() _NOEXCEPT = default;
#else
  __cxx_atomic_base_impl() _NOEXCEPT : __value_() {}
#endif // _LIBCPP_CXX03_LANG

  _LIBCPP_CONSTEXPR __cxx_atomic_base_impl(_Tp __value) _NOEXCEPT : __value_(std::__clear_padding_if_needed(__value)) {}

  void __atomic_init(_Tp __val) volatile {
    std::__clear_padding_if_needed(__val);
    __atomic_assign_volatile(__value_, __val);
  }

  void __atomic_init(_Tp __val) {
    std::__clear_padding_if_needed(__val);
    __value_ = __val;
  }

  void __atomic_store(_Tp __val, memory_order __order) volatile {
    std::__clear_padding_if_needed(__val);
    ::__atomic_store(std::addressof(__value_), std::addressof(__val), std::__to_gcc_order(__order));
  }

  void __atomic_store(_Tp __val, memory_order __order) {
    std::__clear_padding_if_needed(__val);
    ::__atomic_store(std::addressof(__value_), std::addressof(__val), std::__to_gcc_order(__order));
  }

  _Tp __atomic_load(memory_order __order) const volatile {
    _Tp __ret;
    ::__atomic_load(std::addressof(__value_), std::addressof(__ret), std::__to_gcc_order(__order));
    return __ret;
  }

  _Tp __atomic_load(memory_order __order) const {
    _Tp __ret;
    ::__atomic_load(std::addressof(__value_), std::addressof(__ret), std::__to_gcc_order(__order));
    return __ret;
  }

  _Tp __atomic_exchange(_Tp __value, memory_order __order) volatile {
    _Tp __ret;
    std::__clear_padding_if_needed(__value);
    ::__atomic_exchange(
        std::addressof(__value_), std::addressof(__value), std::addressof(__ret), std::__to_gcc_order(__order));
    return __ret;
  }

  _Tp __atomic_exchange(_Tp __value, memory_order __order) {
    _Tp __ret;
    std::__clear_padding_if_needed(__value);
    ::__atomic_exchange(
        std::addressof(__value_), std::addressof(__value), std::addressof(__ret), std::__to_gcc_order(__order));
    return __ret;
  }

  bool __atomic_compare_exchange_strong(
      _Tp* __expected, _Tp __value, memory_order __success, memory_order __failure) volatile {
    return std::__atomic_cas_with_clear_padding(
        __expected, __value, [this, __success, __failure](_Tp* __expected_or_copy, _Tp __value_maybe_padding_cleared) {
          return ::__atomic_compare_exchange(
              std::addressof(this->__value_),
              __expected_or_copy,
              std::addressof(__value_maybe_padding_cleared),
              false,
              std::__to_gcc_order(__success),
              std::__to_gcc_failure_order(__failure));
        });
  }

  bool __atomic_compare_exchange_strong(_Tp* __expected, _Tp __value, memory_order __success, memory_order __failure) {
    return std::__atomic_cas_with_clear_padding(
        __expected, __value, [this, __success, __failure](_Tp* __expected_or_copy, _Tp __value_maybe_padding_cleared) {
          return ::__atomic_compare_exchange(
              std::addressof(this->__value_),
              __expected_or_copy,
              std::addressof(__value_maybe_padding_cleared),
              false,
              std::__to_gcc_order(__success),
              std::__to_gcc_failure_order(__failure));
        });
  }

  bool __atomic_compare_exchange_weak(
      _Tp* __expected, _Tp __value, memory_order __success, memory_order __failure) volatile {
    return std::__atomic_cas_with_clear_padding(
        __expected, __value, [this, __success, __failure](_Tp* __expected_or_copy, _Tp __value_maybe_padding_cleared) {
          return ::__atomic_compare_exchange(
              std::addressof(this->__value_),
              __expected_or_copy,
              std::addressof(__value_maybe_padding_cleared),
              true,
              std::__to_gcc_order(__success),
              std::__to_gcc_failure_order(__failure));
        });
  }

  bool __atomic_compare_exchange_weak(_Tp* __expected, _Tp __value, memory_order __success, memory_order __failure) {
    return std::__atomic_cas_with_clear_padding(
        __expected, __value, [this, __success, __failure](_Tp* __expected_or_copy, _Tp __value_maybe_padding_cleared) {
          return ::__atomic_compare_exchange(
              std::addressof(this->__value_),
              __expected_or_copy,
              std::addressof(__value_maybe_padding_cleared),
              true,
              std::__to_gcc_order(__success),
              std::__to_gcc_failure_order(__failure));
        });
  }

  template <typename _Td>
  _Tp __atomic_fetch_add(_Td __delta, memory_order __order) volatile {
    return ::__atomic_fetch_add(
        std::addressof(__value_), __delta * __skip_amt<_Tp>::value, std::__to_gcc_order(__order));
  }

  template <typename _Td>
  _Tp __atomic_fetch_add(_Td __delta, memory_order __order) {
    return ::__atomic_fetch_add(
        std::addressof(__value_), __delta * __skip_amt<_Tp>::value, std::__to_gcc_order(__order));
  }

  template <typename _Td>
  _Tp __atomic_fetch_sub(_Td __delta, memory_order __order) volatile {
    return ::__atomic_fetch_sub(
        std::addressof(__value_), __delta * __skip_amt<_Tp>::value, std::__to_gcc_order(__order));
  }

  template <typename _Td>
  _Tp __atomic_fetch_sub(_Td __delta, memory_order __order) {
    return ::__atomic_fetch_sub(
        std::addressof(__value_), __delta * __skip_amt<_Tp>::value, std::__to_gcc_order(__order));
  }

  _Tp __atomic_fetch_and(_Tp __pattern, memory_order __order) volatile {
    return ::__atomic_fetch_and(std::addressof(__value_), __pattern, std::__to_gcc_order(__order));
  }

  _Tp __atomic_fetch_and(_Tp __pattern, memory_order __order) {
    return ::__atomic_fetch_and(std::addressof(__value_), __pattern, std::__to_gcc_order(__order));
  }

  _Tp __atomic_fetch_or(_Tp __pattern, memory_order __order) volatile {
    return ::__atomic_fetch_or(std::addressof(__value_), __pattern, std::__to_gcc_order(__order));
  }

  _Tp __atomic_fetch_or(_Tp __pattern, memory_order __order) {
    return ::__atomic_fetch_or(std::addressof(__value_), __pattern, std::__to_gcc_order(__order));
  }

  _Tp __atomic_fetch_xor(_Tp __pattern, memory_order __order) volatile {
    return ::__atomic_fetch_xor(std::addressof(__value_), __pattern, std::__to_gcc_order(__order));
  }

  _Tp __atomic_fetch_xor(_Tp __pattern, memory_order __order) {
    return ::__atomic_fetch_xor(std::addressof(__value_), __pattern, std::__to_gcc_order(__order));
  }

  _Tp __atomic_fetch_max(_Tp __val, memory_order __order) volatile {
#if __has_builtin(__atomic_fetch_max)
    return ::__atomic_fetch_max(std::addressof(__value_), __val, std::__to_gcc_order(__order));
#else
    _Tp __ret = this->__atomic_load(memory_order_relaxed);
    _Tp __value;
    do {
      __value = __ret > __val ? __ret : __val;
    } while (!this->__atomic_compare_exchange_weak(std::addressof(__ret), __value, __order, memory_order_relaxed));
    return __ret;
#endif
  }

  _Tp __atomic_fetch_max(_Tp __val, memory_order __order) {
#if __has_builtin(__atomic_fetch_max)
    return ::__atomic_fetch_max(std::addressof(__value_), __val, std::__to_gcc_order(__order));
#else
    _Tp __ret = this->__atomic_load(memory_order_relaxed);
    _Tp __value;
    do {
      __value = __ret > __val ? __ret : __val;
    } while (!this->__atomic_compare_exchange_weak(std::addressof(__ret), __value, __order, memory_order_relaxed));
    return __ret;
#endif
  }

  _Tp __atomic_fetch_min(_Tp __val, memory_order __order) volatile {
#if __has_builtin(__atomic_fetch_min)
    return ::__atomic_fetch_min(std::addressof(__value_), __val, std::__to_gcc_order(__order));
#else
    _Tp __ret = this->__atomic_load(memory_order_relaxed);
    _Tp __value;
    do {
      __value = __ret < __val ? __ret : __val;
    } while (!this->__atomic_compare_exchange_weak(std::addressof(__ret), __value, __order, memory_order_relaxed));
    return __ret;
#endif
  }

  _Tp __atomic_fetch_min(_Tp __val, memory_order __order) {
#if __has_builtin(__atomic_fetch_min)
    return ::__atomic_fetch_min(std::addressof(__value_), __val, std::__to_gcc_order(__order));
#else
    _Tp __ret = this->__atomic_load(memory_order_relaxed);
    _Tp __value;
    do {
      __value = __ret < __val ? __ret : __val;
    } while (!this->__atomic_compare_exchange_weak(std::addressof(__ret), __value, __order, memory_order_relaxed));
    return __ret;
#endif
  }

private:
  // [atomics.types.generic]p1 guarantees _Tp is trivially copyable. Because
  // the default operator= in an object is not volatile, a byte-by-byte copy
  // is required.
  template <typename _Tv, __enable_if_t<is_assignable<_Tp&, _Tv>::value, int> = 0>
  static void __atomic_assign_volatile(_Tp volatile& __value_, _Tv volatile const& __val) {
    volatile char* __to         = reinterpret_cast<volatile char*>(std::addressof(__value_));
    volatile char* __end        = __to + sizeof(_Tp);
    volatile const char* __from = reinterpret_cast<volatile const char*>(std::addressof(__val));
    while (__to != __end)
      *__to++ = *__from++;
  }

public:
  _Tp __value_;
};

inline void __cxx_atomic_thread_fence(memory_order __order) { __atomic_thread_fence(std::__to_gcc_order(__order)); }

inline void __cxx_atomic_signal_fence(memory_order __order) { __atomic_signal_fence(std::__to_gcc_order(__order)); }

#define __cxx_atomic_is_lock_free(__s) __atomic_is_lock_free(__s, 0)

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___ATOMIC_SUPPORT_GCC_H
