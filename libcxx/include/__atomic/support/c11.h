//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___ATOMIC_SUPPORT_C11_H
#define _LIBCPP___ATOMIC_SUPPORT_C11_H

#include <__atomic/clear_padding.h>
#include <__atomic/memory_order.h>
#include <__config>
#include <__cstddef/ptrdiff_t.h>
#include <__memory/addressof.h>
#include <__type_traits/conditional.h>
#include <__type_traits/enable_if.h>
#include <__type_traits/is_pointer.h>
#include <__type_traits/remove_const.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

//
// This file implements support for C11-style atomics
//

_LIBCPP_BEGIN_NAMESPACE_STD

template <typename _Tp>
struct __cxx_atomic_base_impl {
#ifndef _LIBCPP_CXX03_LANG
  __cxx_atomic_base_impl() _NOEXCEPT = default;
#else
  __cxx_atomic_base_impl() _NOEXCEPT : __atomic_value_() {}
#endif // _LIBCPP_CXX03_LANG

  _LIBCPP_CONSTEXPR explicit __cxx_atomic_base_impl(_Tp __value) _NOEXCEPT
      : __atomic_value_(std::__clear_padding_if_needed(__value)) {}

  void __atomic_init(_Tp __val) volatile _NOEXCEPT {
    std::__clear_padding_if_needed(__val);
    __c11_atomic_init(std::addressof(__atomic_value_), __val);
  }

  void __atomic_init(_Tp __val) _NOEXCEPT {
    std::__clear_padding_if_needed(__val);
    __c11_atomic_init(std::addressof(__atomic_value_), __val);
  }

  void __atomic_store(_Tp __val, memory_order __order) volatile _NOEXCEPT {
    std::__clear_padding_if_needed(__val);
    __c11_atomic_store(std::addressof(__atomic_value_), __val, static_cast<__memory_order_underlying_t>(__order));
  }

  void __atomic_store(_Tp __val, memory_order __order) _NOEXCEPT {
    std::__clear_padding_if_needed(__val);
    __c11_atomic_store(std::addressof(__atomic_value_), __val, static_cast<__memory_order_underlying_t>(__order));
  }

  _Tp __atomic_load(memory_order __order) const volatile _NOEXCEPT {
    using __ptr_type = __remove_const_t<decltype(__atomic_value_)>*;
    return __c11_atomic_load(
        const_cast<__ptr_type>(std::addressof(__atomic_value_)), static_cast<__memory_order_underlying_t>(__order));
  }

  _Tp __atomic_load(memory_order __order) const _NOEXCEPT {
    using __ptr_type = __remove_const_t<decltype(__atomic_value_)>*;
    return __c11_atomic_load(
        const_cast<__ptr_type>(std::addressof(__atomic_value_)), static_cast<__memory_order_underlying_t>(__order));
  }

  _Tp __atomic_exchange(_Tp __value, memory_order __order) volatile _NOEXCEPT {
    std::__clear_padding_if_needed(__value);
    return __c11_atomic_exchange(
        std::addressof(__atomic_value_), __value, static_cast<__memory_order_underlying_t>(__order));
  }

  _Tp __atomic_exchange(_Tp __value, memory_order __order) _NOEXCEPT {
    std::__clear_padding_if_needed(__value);
    return __c11_atomic_exchange(
        std::addressof(__atomic_value_), __value, static_cast<__memory_order_underlying_t>(__order));
  }

  bool __atomic_compare_exchange_strong(
      _Tp* __expected, _Tp __value, memory_order __success, memory_order __failure) volatile _NOEXCEPT {
    return std::__atomic_cas_with_clear_padding(
        __expected,
        __value,
        // Use always_inline since Clang fails to inline the lambda, even when the memory order is a constant
        [this, __success, __failure](_Tp* __expected_or_copy, _Tp __value_maybe_padding_cleared) _LIBCPP_ALWAYS_INLINE {
          return __c11_atomic_compare_exchange_strong(
              std::addressof(this->__atomic_value_),
              __expected_or_copy,
              __value_maybe_padding_cleared,
              static_cast<__memory_order_underlying_t>(__success),
              static_cast<__memory_order_underlying_t>(__to_failure_order(__failure)));
        });
  }

  bool __atomic_compare_exchange_strong(
      _Tp* __expected, _Tp __value, memory_order __success, memory_order __failure) _NOEXCEPT {
    return std::__atomic_cas_with_clear_padding(
        __expected,
        __value,
        [this, __success, __failure](_Tp* __expected_or_copy, _Tp __value_maybe_padding_cleared) _LIBCPP_ALWAYS_INLINE {
          return __c11_atomic_compare_exchange_strong(
              std::addressof(this->__atomic_value_),
              __expected_or_copy,
              __value_maybe_padding_cleared,
              static_cast<__memory_order_underlying_t>(__success),
              static_cast<__memory_order_underlying_t>(__to_failure_order(__failure)));
        });
  }

  bool __atomic_compare_exchange_weak(
      _Tp* __expected, _Tp __value, memory_order __success, memory_order __failure) volatile _NOEXCEPT {
    return std::__atomic_cas_with_clear_padding(
        __expected,
        __value,
        [this, __success, __failure](_Tp* __expected_or_copy, _Tp __value_maybe_padding_cleared) _LIBCPP_ALWAYS_INLINE {
          return __c11_atomic_compare_exchange_weak(
              std::addressof(this->__atomic_value_),
              __expected_or_copy,
              __value_maybe_padding_cleared,
              static_cast<__memory_order_underlying_t>(__success),
              static_cast<__memory_order_underlying_t>(__to_failure_order(__failure)));
        });
  }

  bool __atomic_compare_exchange_weak(
      _Tp* __expected, _Tp __value, memory_order __success, memory_order __failure) _NOEXCEPT {
    return std::__atomic_cas_with_clear_padding(
        __expected,
        __value,
        [this, __success, __failure](_Tp* __expected_or_copy, _Tp __value_maybe_padding_cleared) _LIBCPP_ALWAYS_INLINE {
          return __c11_atomic_compare_exchange_weak(
              std::addressof(this->__atomic_value_),
              __expected_or_copy,
              __value_maybe_padding_cleared,
              static_cast<__memory_order_underlying_t>(__success),
              static_cast<__memory_order_underlying_t>(__to_failure_order(__failure)));
        });
  }

  _Tp __atomic_fetch_add(conditional<is_pointer<_Tp>::value, ptrdiff_t, _Tp>::type __delta,
                         memory_order __order) volatile _NOEXCEPT {
    return __c11_atomic_fetch_add(
        std::addressof(__atomic_value_), __delta, static_cast<__memory_order_underlying_t>(__order));
  }

  _Tp __atomic_fetch_add(conditional<is_pointer<_Tp>::value, ptrdiff_t, _Tp>::type __delta,
                         memory_order __order) _NOEXCEPT {
    return __c11_atomic_fetch_add(
        std::addressof(__atomic_value_), __delta, static_cast<__memory_order_underlying_t>(__order));
  }

  _Tp __atomic_fetch_sub(conditional<is_pointer<_Tp>::value, ptrdiff_t, _Tp>::type __delta,
                         memory_order __order) volatile _NOEXCEPT {
    return __c11_atomic_fetch_sub(
        std::addressof(__atomic_value_), __delta, static_cast<__memory_order_underlying_t>(__order));
  }

  _Tp __atomic_fetch_sub(conditional<is_pointer<_Tp>::value, ptrdiff_t, _Tp>::type __delta,
                         memory_order __order) _NOEXCEPT {
    return __c11_atomic_fetch_sub(
        std::addressof(__atomic_value_), __delta, static_cast<__memory_order_underlying_t>(__order));
  }

  _Tp __atomic_fetch_and(_Tp __pattern, memory_order __order) volatile _NOEXCEPT {
    return __c11_atomic_fetch_and(
        std::addressof(__atomic_value_), __pattern, static_cast<__memory_order_underlying_t>(__order));
  }

  _Tp __atomic_fetch_and(_Tp __pattern, memory_order __order) _NOEXCEPT {
    return __c11_atomic_fetch_and(
        std::addressof(__atomic_value_), __pattern, static_cast<__memory_order_underlying_t>(__order));
  }

  _Tp __atomic_fetch_or(_Tp __pattern, memory_order __order) volatile _NOEXCEPT {
    return __c11_atomic_fetch_or(
        std::addressof(__atomic_value_), __pattern, static_cast<__memory_order_underlying_t>(__order));
  }

  _Tp __atomic_fetch_or(_Tp __pattern, memory_order __order) _NOEXCEPT {
    return __c11_atomic_fetch_or(
        std::addressof(__atomic_value_), __pattern, static_cast<__memory_order_underlying_t>(__order));
  }

  _Tp __atomic_fetch_xor(_Tp __pattern, memory_order __order) volatile _NOEXCEPT {
    return __c11_atomic_fetch_xor(
        std::addressof(__atomic_value_), __pattern, static_cast<__memory_order_underlying_t>(__order));
  }

  _Tp __atomic_fetch_xor(_Tp __pattern, memory_order __order) _NOEXCEPT {
    return __c11_atomic_fetch_xor(
        std::addressof(__atomic_value_), __pattern, static_cast<__memory_order_underlying_t>(__order));
  }

  _Tp __atomic_fetch_max(_Tp __val, memory_order __order) volatile _NOEXCEPT {
    return __atomic_fetch_max_impl(*this, __val, __order);
  }

  _Tp __atomic_fetch_max(_Tp __val, memory_order __order) _NOEXCEPT {
    return __atomic_fetch_max_impl(*this, __val, __order);
  }

  _Tp __atomic_fetch_min(_Tp __val, memory_order __order) volatile _NOEXCEPT {
    return __atomic_fetch_min_impl(*this, __val, __order);
  }

  _Tp __atomic_fetch_min(_Tp __val, memory_order __order) _NOEXCEPT {
    return __atomic_fetch_min_impl(*this, __val, __order);
  }

private:
  inline static _LIBCPP_CONSTEXPR memory_order __to_failure_order(memory_order __order) {
    // Avoid switch statement to make this a constexpr.
    return __order == memory_order_release
             ? memory_order_relaxed
             : (__order == memory_order_acq_rel ? memory_order_acquire : __order);
  }

  // Clang's __c11_atomic_fetch_max/min builtins do not accept pointer
  // arguments, so dispatch to a CAS loop for pointer types.
  // TODO: Use the builtin for pointer types once Clang accepts them.
  template <class _Self, class _Up = _Tp, __enable_if_t<!is_pointer<_Up>::value, int> = 0>
  static _Tp __atomic_fetch_max_impl(_Self&& __self, _Tp __val, memory_order __order) _NOEXCEPT {
    return __c11_atomic_fetch_max(
        std::addressof(__self.__atomic_value_), __val, static_cast<__memory_order_underlying_t>(__order));
  }

  template <class _Self, class _Up = _Tp, __enable_if_t<is_pointer<_Up>::value, int> = 0>
  static _Tp __atomic_fetch_max_impl(_Self&& __self, _Tp __val, memory_order __order) _NOEXCEPT {
    _Tp __ret = __self.__atomic_load(memory_order_relaxed);
    _Tp __value;
    do {
      __value = __ret > __val ? __ret : __val;
    } while (!__self.__atomic_compare_exchange_weak(std::addressof(__ret), __value, __order, memory_order_relaxed));
    return __ret;
  }

  template <class _Self, class _Up = _Tp, __enable_if_t<!is_pointer<_Up>::value, int> = 0>
  static _Tp __atomic_fetch_min_impl(_Self&& __self, _Tp __val, memory_order __order) _NOEXCEPT {
    return __c11_atomic_fetch_min(
        std::addressof(__self.__atomic_value_), __val, static_cast<__memory_order_underlying_t>(__order));
  }

  template <class _Self, class _Up = _Tp, __enable_if_t<is_pointer<_Up>::value, int> = 0>
  static _Tp __atomic_fetch_min_impl(_Self&& __self, _Tp __val, memory_order __order) _NOEXCEPT {
    _Tp __ret = __self.__atomic_load(memory_order_relaxed);
    _Tp __value;
    do {
      __value = __ret < __val ? __ret : __val;
    } while (!__self.__atomic_compare_exchange_weak(std::addressof(__ret), __value, __order, memory_order_relaxed));
    return __ret;
  }

  // backward compatibility: it was public

public:
  _Atomic(_Tp) __atomic_value_;
};

#define __cxx_atomic_is_lock_free(__s) __c11_atomic_is_lock_free(__s)

inline void __cxx_atomic_thread_fence(memory_order __order) _NOEXCEPT {
  __c11_atomic_thread_fence(static_cast<__memory_order_underlying_t>(__order));
}

inline void __cxx_atomic_signal_fence(memory_order __order) _NOEXCEPT {
  __c11_atomic_signal_fence(static_cast<__memory_order_underlying_t>(__order));
}

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___ATOMIC_SUPPORT_C11_H
