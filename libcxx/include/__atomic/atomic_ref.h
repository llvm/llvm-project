// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//                        Kokkos v. 4.0
//       Copyright (2022) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
//===---------------------------------------------------------------------===//

#ifndef _LIBCPP___ATOMIC_ATOMIC_REF_H
#define _LIBCPP___ATOMIC_ATOMIC_REF_H

#include <__assert>
#include <__atomic/check_memory_order.h>
#include <__atomic/to_gcc_order.h>
#include <__config>
#include <__memory/addressof.h>
#include <__type_traits/is_floating_point.h>
#include <__type_traits/is_integral.h>
#include <__type_traits/is_same.h>
#include <__type_traits/is_trivially_copyable.h>
#include <cstddef>
#include <cstdint>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_PUSH_MACROS
#include <__undef_macros>

_LIBCPP_BEGIN_NAMESPACE_STD

#if _LIBCPP_STD_VER >= 20

template <class _Tp, bool = is_integral_v<_Tp> && !is_same_v<_Tp, bool>, bool = is_floating_point_v<_Tp>>
struct __atomic_ref_base {
  _Tp* __ptr_;

  using value_type = _Tp;

  static constexpr size_t required_alignment = alignof(_Tp);

  static constexpr bool is_always_lock_free = __atomic_always_lock_free(sizeof(_Tp), 0);

  _LIBCPP_HIDE_FROM_ABI bool is_lock_free() const noexcept { return __atomic_is_lock_free(sizeof(_Tp), 0); }

  _LIBCPP_HIDE_FROM_ABI void store(_Tp __desired, memory_order __order = memory_order::seq_cst) const noexcept
      _LIBCPP_CHECK_STORE_MEMORY_ORDER(__order) {
    _LIBCPP_ASSERT_UNCATEGORIZED(
        __order == memory_order::relaxed || __order == memory_order::release || __order == memory_order::seq_cst,
        "memory order argument to atomic store operation is invalid");
    __atomic_store(__ptr_, std::addressof(__desired), __to_gcc_order(__order));
  }

  _LIBCPP_HIDE_FROM_ABI _Tp operator=(_Tp __desired) const noexcept {
    store(__desired);
    return __desired;
  }

  _LIBCPP_HIDE_FROM_ABI _Tp load(memory_order __order = memory_order::seq_cst) const noexcept
      _LIBCPP_CHECK_LOAD_MEMORY_ORDER(__order) {
    _LIBCPP_ASSERT_UNCATEGORIZED(
        __order == memory_order::relaxed || __order == memory_order::consume || __order == memory_order::acquire ||
            __order == memory_order::seq_cst,
        "memory order argument to atomic load operation is invalid");
    alignas(_Tp) unsigned char __mem[sizeof(_Tp)];
    auto* __ret = reinterpret_cast<_Tp*>(__mem);
    __atomic_load(__ptr_, __ret, __to_gcc_order(__order));
    return *__ret;
  }

  _LIBCPP_HIDE_FROM_ABI operator _Tp() const noexcept { return load(); }

  _LIBCPP_HIDE_FROM_ABI _Tp exchange(_Tp __desired, memory_order __order = memory_order::seq_cst) const noexcept {
    alignas(_Tp) unsigned char __mem[sizeof(_Tp)];
    auto* __ret = reinterpret_cast<_Tp*>(__mem);
    __atomic_exchange(__ptr_, std::addressof(__desired), __ret, __to_gcc_order(__order));
    return *__ret;
  }
  _LIBCPP_HIDE_FROM_ABI bool
  compare_exchange_weak(_Tp& __expected, _Tp __desired, memory_order __success, memory_order __failure) const noexcept
      _LIBCPP_CHECK_EXCHANGE_MEMORY_ORDER(__success, __failure) {
    _LIBCPP_ASSERT_UNCATEGORIZED(
        __failure == memory_order::relaxed || __failure == memory_order::consume ||
            __failure == memory_order::acquire || __failure == memory_order::seq_cst,
        "failure memory order argument to weak atomic compare-and-exchange operation is invalid");
    return __atomic_compare_exchange(
        __ptr_,
        std::addressof(__expected),
        std::addressof(__desired),
        true,
        __to_gcc_order(__success),
        __to_gcc_order(__failure));
  }
  _LIBCPP_HIDE_FROM_ABI bool
  compare_exchange_strong(_Tp& __expected, _Tp __desired, memory_order __success, memory_order __failure) const noexcept
      _LIBCPP_CHECK_EXCHANGE_MEMORY_ORDER(__success, __failure) {
    _LIBCPP_ASSERT_UNCATEGORIZED(
        __failure == memory_order::relaxed || __failure == memory_order::consume ||
            __failure == memory_order::acquire || __failure == memory_order::seq_cst,
        "failure memory order argument to strong atomic compare-and-exchange operation is invalid");
    return __atomic_compare_exchange(
        __ptr_,
        std::addressof(__expected),
        std::addressof(__desired),
        false,
        __to_gcc_order(__success),
        __to_gcc_order(__failure));
  }

  _LIBCPP_HIDE_FROM_ABI bool
  compare_exchange_weak(_Tp& __expected, _Tp __desired, memory_order __order = memory_order::seq_cst) const noexcept {
    return __atomic_compare_exchange(
        __ptr_,
        std::addressof(__expected),
        std::addressof(__desired),
        true,
        __to_gcc_order(__order),
        __to_gcc_failure_order(__order));
  }
  _LIBCPP_HIDE_FROM_ABI bool
  compare_exchange_strong(_Tp& __expected, _Tp __desired, memory_order __order = memory_order::seq_cst) const noexcept {
    return __atomic_compare_exchange(
        __ptr_,
        std::addressof(__expected),
        std::addressof(__desired),
        false,
        __to_gcc_order(__order),
        __to_gcc_failure_order(__order));
  }

  _LIBCPP_HIDE_FROM_ABI void wait(_Tp __old, memory_order __order = memory_order::seq_cst) const noexcept
      _LIBCPP_CHECK_WAIT_MEMORY_ORDER(__order) {
    _LIBCPP_ASSERT_UNCATEGORIZED(
        __order == memory_order::relaxed || __order == memory_order::consume || __order == memory_order::acquire ||
            __order == memory_order::seq_cst,
        "memory order argument to atomic wait operation is invalid");
    // FIXME
    (void)__old;
    (void)__order;
  }
  _LIBCPP_HIDE_FROM_ABI void notify_one() const noexcept {
    // FIXME
  }
  _LIBCPP_HIDE_FROM_ABI void notify_all() const noexcept {
    // FIXME
  }

  _LIBCPP_HIDE_FROM_ABI __atomic_ref_base(_Tp& __obj) : __ptr_(&__obj) {}
};

template <class _Tp>
struct __atomic_ref_base<_Tp, /*_IsIntegral=*/true, /*_IsFloatingPoint=*/false>
    : public __atomic_ref_base<_Tp, false, false> {
  using __base = __atomic_ref_base<_Tp, false, false>;

  using difference_type = __base::value_type;

  _LIBCPP_HIDE_FROM_ABI __atomic_ref_base(_Tp& __obj) : __base(__obj) {}

  _LIBCPP_HIDE_FROM_ABI _Tp operator=(_Tp __desired) const noexcept { return __base::operator=(__desired); }

  _LIBCPP_HIDE_FROM_ABI _Tp fetch_add(_Tp __arg, memory_order __order = memory_order_seq_cst) const noexcept {
    return __atomic_fetch_add(this->__ptr_, __arg, __to_gcc_order(__order));
  }
  _LIBCPP_HIDE_FROM_ABI _Tp fetch_sub(_Tp __arg, memory_order __order = memory_order_seq_cst) const noexcept {
    return __atomic_fetch_sub(this->__ptr_, __arg, __to_gcc_order(__order));
  }
  _LIBCPP_HIDE_FROM_ABI _Tp fetch_and(_Tp __arg, memory_order __order = memory_order_seq_cst) const noexcept {
    return __atomic_fetch_and(this->__ptr_, __arg, __to_gcc_order(__order));
  }
  _LIBCPP_HIDE_FROM_ABI _Tp fetch_or(_Tp __arg, memory_order __order = memory_order_seq_cst) const noexcept {
    return __atomic_fetch_or(this->__ptr_, __arg, __to_gcc_order(__order));
  }
  _LIBCPP_HIDE_FROM_ABI _Tp fetch_xor(_Tp __arg, memory_order __order = memory_order_seq_cst) const noexcept {
    return __atomic_fetch_xor(this->__ptr_, __arg, __to_gcc_order(__order));
  }

  _LIBCPP_HIDE_FROM_ABI _Tp operator++(int) const noexcept { return fetch_add(_Tp(1)); }
  _LIBCPP_HIDE_FROM_ABI _Tp operator--(int) const noexcept { return fetch_sub(_Tp(1)); }
  _LIBCPP_HIDE_FROM_ABI _Tp operator++() const noexcept { return fetch_add(_Tp(1)) + _Tp(1); }
  _LIBCPP_HIDE_FROM_ABI _Tp operator--() const noexcept { return fetch_sub(_Tp(1)) - _Tp(1); }
  _LIBCPP_HIDE_FROM_ABI _Tp operator+=(_Tp __arg) const noexcept { return fetch_add(__arg) + __arg; }
  _LIBCPP_HIDE_FROM_ABI _Tp operator-=(_Tp __arg) const noexcept { return fetch_sub(__arg) - __arg; }
  _LIBCPP_HIDE_FROM_ABI _Tp operator&=(_Tp __arg) const noexcept { return fetch_and(__arg) & __arg; }
  _LIBCPP_HIDE_FROM_ABI _Tp operator|=(_Tp __arg) const noexcept { return fetch_or(__arg) | __arg; }
  _LIBCPP_HIDE_FROM_ABI _Tp operator^=(_Tp __arg) const noexcept { return fetch_xor(__arg) ^ __arg; }
};

template <class _Tp>
struct __atomic_ref_base<_Tp, /*_IsIntegral=*/false, /*_IsFloatingPoint=*/true>
    : public __atomic_ref_base<_Tp, false, false> {
  using __base = __atomic_ref_base<_Tp, false, false>;

  using difference_type = __base::value_type;

  _LIBCPP_HIDE_FROM_ABI __atomic_ref_base(_Tp& __obj) : __base(__obj) {}

  _LIBCPP_HIDE_FROM_ABI _Tp operator=(_Tp __desired) const noexcept { return __base::operator=(__desired); }

  _LIBCPP_HIDE_FROM_ABI _Tp fetch_add(_Tp __arg, memory_order __order = memory_order_seq_cst) const noexcept {
    _Tp __old = this->load(memory_order_relaxed);
    _Tp __new = __old + __arg;
    while (!this->compare_exchange_weak(__old, __new, __order, memory_order_relaxed)) {
      __new = __old + __arg;
    }
    return __old;
  }
  _LIBCPP_HIDE_FROM_ABI _Tp fetch_sub(_Tp __arg, memory_order __order = memory_order_seq_cst) const noexcept {
    _Tp __old = this->load(memory_order_relaxed);
    _Tp __new = __old - __arg;
    while (!this->compare_exchange_weak(__old, __new, __order, memory_order_relaxed)) {
      __new = __old - __arg;
    }
    return __old;
  }

  _LIBCPP_HIDE_FROM_ABI _Tp operator+=(_Tp __arg) const noexcept { return fetch_add(__arg) + __arg; }
  _LIBCPP_HIDE_FROM_ABI _Tp operator-=(_Tp __arg) const noexcept { return fetch_sub(__arg) - __arg; }
};

template <class _Tp>
struct atomic_ref : public __atomic_ref_base<_Tp> {
  static_assert(is_trivially_copyable_v<_Tp>, "std::atomic_ref<T> requires that 'T' be a trivially copyable type");

  using __base = __atomic_ref_base<_Tp>;

  _LIBCPP_HIDE_FROM_ABI explicit atomic_ref(_Tp& __obj) : __base(__obj) {
    _LIBCPP_ASSERT_UNCATEGORIZED((uintptr_t)addressof(__obj) % __base::required_alignment == 0,
                                 "atomic_ref ctor: referenced object must be aligned to required_alignment");
  }

  _LIBCPP_HIDE_FROM_ABI atomic_ref(const atomic_ref&) noexcept = default;

  _LIBCPP_HIDE_FROM_ABI _Tp operator=(_Tp __desired) const noexcept { return __base::operator=(__desired); }

  atomic_ref& operator=(const atomic_ref&) = delete;
};

template <class _Tp>
struct atomic_ref<_Tp*> : public __atomic_ref_base<_Tp*> {
  using __base = __atomic_ref_base<_Tp*>;

  using difference_type = ptrdiff_t;

  _LIBCPP_HIDE_FROM_ABI _Tp* fetch_add(ptrdiff_t __arg, memory_order __order = memory_order_seq_cst) const noexcept {
    return __atomic_fetch_add(this->__ptr_, __arg * sizeof(_Tp), __to_gcc_order(__order));
  }
  _LIBCPP_HIDE_FROM_ABI _Tp* fetch_sub(ptrdiff_t __arg, memory_order __order = memory_order_seq_cst) const noexcept {
    return __atomic_fetch_sub(this->__ptr_, __arg * sizeof(_Tp), __to_gcc_order(__order));
  }

  _LIBCPP_HIDE_FROM_ABI _Tp* operator++(int) const noexcept { return fetch_add(1); }
  _LIBCPP_HIDE_FROM_ABI _Tp* operator--(int) const noexcept { return fetch_sub(1); }
  _LIBCPP_HIDE_FROM_ABI _Tp* operator++() const noexcept { return fetch_add(1) + 1; }
  _LIBCPP_HIDE_FROM_ABI _Tp* operator--() const noexcept { return fetch_sub(1) - 1; }
  _LIBCPP_HIDE_FROM_ABI _Tp* operator+=(ptrdiff_t __arg) const noexcept { return fetch_add(__arg) + __arg; }
  _LIBCPP_HIDE_FROM_ABI _Tp* operator-=(ptrdiff_t __arg) const noexcept { return fetch_sub(__arg) - __arg; }

  _LIBCPP_HIDE_FROM_ABI explicit atomic_ref(_Tp*& __ptr) : __base(__ptr) {}

  _LIBCPP_HIDE_FROM_ABI _Tp* operator=(_Tp* __desired) const noexcept { return __base::operator=(__desired); }

  atomic_ref& operator=(const atomic_ref&) = delete;
};

#endif // _LIBCPP_STD_VER >= 20

_LIBCPP_END_NAMESPACE_STD

_LIBCPP_POP_MACROS

#endif // _LIBCPP__ATOMIC_ATOMIC_REF_H
