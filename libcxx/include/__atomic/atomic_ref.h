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
#include <__atomic/atomic_sync.h>
#include <__atomic/check_memory_order.h>
#include <__atomic/memory_order.h>
#include <__atomic/to_gcc_order.h>
#include <__concepts/arithmetic.h>
#include <__concepts/same_as.h>
#include <__config>
#include <__cstddef/byte.h>
#include <__cstddef/ptrdiff_t.h>
#include <__memory/addressof.h>
#include <__type_traits/has_unique_object_representation.h>
#include <__type_traits/is_const.h>
#include <__type_traits/is_pointer.h>
#include <__type_traits/is_trivially_copyable.h>
#include <__type_traits/is_volatile.h>
#include <__type_traits/remove_cv.h>
#include <__type_traits/remove_pointer.h>
#include <cstdint>
#include <cstring>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_PUSH_MACROS
#include <__undef_macros>

_LIBCPP_BEGIN_NAMESPACE_STD

#if _LIBCPP_STD_VER >= 20

// These types are required to make __atomic_is_always_lock_free work across GCC and Clang.
// The purpose of this trick is to make sure that we provide an object with the correct alignment
// to __atomic_is_always_lock_free, since that answer depends on the alignment.
template <size_t _Alignment>
struct __alignment_checker_type {
  alignas(_Alignment) char __data;
};

template <size_t _Alignment>
struct __get_aligner_instance {
  static constexpr __alignment_checker_type<_Alignment> __instance{};
};

template <class _Tp>
struct __atomic_ref_base {
  using value_type = __remove_cv_t<_Tp>;

private:
  _LIBCPP_HIDE_FROM_ABI static value_type* __clear_padding(value_type& __val) noexcept {
    value_type* __ptr = std::addressof(__val);
#  if __has_builtin(__builtin_clear_padding)
    __builtin_clear_padding(__ptr);
#  endif
    return __ptr;
  }

  _LIBCPP_HIDE_FROM_ABI static bool __compare_exchange(
      _Tp* __ptr, _Tp* __expected, _Tp* __desired, bool __is_weak, int __success, int __failure) noexcept {
    if constexpr (
#  if __has_builtin(__builtin_clear_padding)
        has_unique_object_representations_v<_Tp> || floating_point<_Tp>
#  else
        true // NOLINT(readability-simplify-boolean-expr)
#  endif
    ) {
      return __atomic_compare_exchange(__ptr, __expected, __desired, __is_weak, __success, __failure);
    } else { // _Tp has padding bits and __builtin_clear_padding is available
      __clear_padding(*__desired);
      _Tp __copy = *__expected;
      __clear_padding(__copy);
      // The algorithm we use here is basically to perform `__atomic_compare_exchange` on the
      // values until it has either succeeded, or failed because the value representation of the
      // objects involved was different. This is why we loop around __atomic_compare_exchange:
      // we basically loop until its failure is caused by the value representation of the objects
      // being different, not only their object representation.
      while (true) {
        _Tp __prev = __copy;
        if (__atomic_compare_exchange(__ptr, std::addressof(__copy), __desired, __is_weak, __success, __failure)) {
          return true;
        }
        _Tp __curr = __copy;
        if (std::memcmp(__clear_padding(__prev), __clear_padding(__curr), sizeof(_Tp)) != 0) {
          // Value representation without padding bits do not compare equal ->
          // write the current content of *ptr into *expected
          std::memcpy(__expected, std::addressof(__copy), sizeof(_Tp));
          return false;
        }
      }
    }
  }

  friend struct __atomic_waitable_traits<__atomic_ref_base<_Tp>>;

  // require types that are 1, 2, 4, 8, or 16 bytes in length to be aligned to at least their size to be potentially
  // used lock-free
  static constexpr size_t __min_alignment = (sizeof(_Tp) & (sizeof(_Tp) - 1)) || (sizeof(_Tp) > 16) ? 0 : sizeof(_Tp);

public:
  static constexpr size_t required_alignment = alignof(_Tp) > __min_alignment ? alignof(_Tp) : __min_alignment;

  // The __atomic_always_lock_free builtin takes into account the alignment of the pointer if provided,
  // so we create a fake pointer with a suitable alignment when querying it. Note that we are guaranteed
  // that the pointer is going to be aligned properly at runtime because that is a (checked) precondition
  // of atomic_ref's constructor.
  static constexpr bool is_always_lock_free =
      __atomic_always_lock_free(sizeof(_Tp), &__get_aligner_instance<required_alignment>::__instance);

  _LIBCPP_HIDE_FROM_ABI bool is_lock_free() const noexcept { return __atomic_is_lock_free(sizeof(_Tp), __ptr_); }

  _LIBCPP_HIDE_FROM_ABI void store(value_type __desired, memory_order __order = memory_order::seq_cst) const noexcept
    requires(!is_const_v<_Tp>)
  _LIBCPP_CHECK_STORE_MEMORY_ORDER(__order) {
    _LIBCPP_ASSERT_ARGUMENT_WITHIN_DOMAIN(
        __order == memory_order::relaxed || __order == memory_order::release || __order == memory_order::seq_cst,
        "atomic_ref: memory order argument to atomic store operation is invalid");
    __atomic_store(__ptr_, __clear_padding(__desired), std::__to_gcc_order(__order));
  }

  _LIBCPP_HIDE_FROM_ABI value_type operator=(value_type __desired) const noexcept
    requires(!is_const_v<_Tp>)
  {
    store(__desired);
    return __desired;
  }

  _LIBCPP_HIDE_FROM_ABI value_type load(memory_order __order = memory_order::seq_cst) const noexcept
      _LIBCPP_CHECK_LOAD_MEMORY_ORDER(__order) {
    _LIBCPP_ASSERT_ARGUMENT_WITHIN_DOMAIN(
        __order == memory_order::relaxed || __order == memory_order::consume || __order == memory_order::acquire ||
            __order == memory_order::seq_cst,
        "atomic_ref: memory order argument to atomic load operation is invalid");
    alignas(value_type) byte __mem[sizeof(value_type)];
    auto* __ret = reinterpret_cast<value_type*>(__mem);
    __atomic_load(__ptr_, __ret, std::__to_gcc_order(__order));
    return *__ret;
  }

  _LIBCPP_HIDE_FROM_ABI operator value_type() const noexcept { return load(); }

  _LIBCPP_HIDE_FROM_ABI value_type
  exchange(value_type __desired, memory_order __order = memory_order::seq_cst) const noexcept
    requires(!is_const_v<_Tp>)
  {
    alignas(value_type) byte __mem[sizeof(value_type)];
    auto* __ret = reinterpret_cast<value_type*>(__mem);
    __atomic_exchange(__ptr_, __clear_padding(__desired), __ret, std::__to_gcc_order(__order));
    return *__ret;
  }

  _LIBCPP_HIDE_FROM_ABI bool compare_exchange_weak(
      value_type& __expected, value_type __desired, memory_order __success, memory_order __failure) const noexcept
    requires(!is_const_v<_Tp>)
  _LIBCPP_CHECK_EXCHANGE_MEMORY_ORDER(__success, __failure) {
    _LIBCPP_ASSERT_ARGUMENT_WITHIN_DOMAIN(
        __failure == memory_order::relaxed || __failure == memory_order::consume ||
            __failure == memory_order::acquire || __failure == memory_order::seq_cst,
        "atomic_ref: failure memory order argument to weak atomic compare-and-exchange operation is invalid");
    return __compare_exchange(
        __ptr_,
        std::addressof(__expected),
        std::addressof(__desired),
        true,
        std::__to_gcc_order(__success),
        std::__to_gcc_order(__failure));
  }
  _LIBCPP_HIDE_FROM_ABI bool compare_exchange_strong(
      value_type& __expected, value_type __desired, memory_order __success, memory_order __failure) const noexcept
    requires(!is_const_v<_Tp>)
  _LIBCPP_CHECK_EXCHANGE_MEMORY_ORDER(__success, __failure) {
    _LIBCPP_ASSERT_ARGUMENT_WITHIN_DOMAIN(
        __failure == memory_order::relaxed || __failure == memory_order::consume ||
            __failure == memory_order::acquire || __failure == memory_order::seq_cst,
        "atomic_ref: failure memory order argument to strong atomic compare-and-exchange operation is invalid");
    return __compare_exchange(
        __ptr_,
        std::addressof(__expected),
        std::addressof(__desired),
        false,
        std::__to_gcc_order(__success),
        std::__to_gcc_order(__failure));
  }

  _LIBCPP_HIDE_FROM_ABI bool compare_exchange_weak(
      value_type& __expected, value_type __desired, memory_order __order = memory_order::seq_cst) const noexcept
    requires(!is_const_v<_Tp>)
  {
    return __compare_exchange(
        __ptr_,
        std::addressof(__expected),
        std::addressof(__desired),
        true,
        std::__to_gcc_order(__order),
        std::__to_gcc_failure_order(__order));
  }
  _LIBCPP_HIDE_FROM_ABI bool compare_exchange_strong(
      value_type& __expected, value_type __desired, memory_order __order = memory_order::seq_cst) const noexcept
    requires(!is_const_v<_Tp>)
  {
    return __compare_exchange(
        __ptr_,
        std::addressof(__expected),
        std::addressof(__desired),
        false,
        std::__to_gcc_order(__order),
        std::__to_gcc_failure_order(__order));
  }

  _LIBCPP_HIDE_FROM_ABI void wait(value_type __old, memory_order __order = memory_order::seq_cst) const noexcept
      _LIBCPP_CHECK_WAIT_MEMORY_ORDER(__order) {
    _LIBCPP_ASSERT_ARGUMENT_WITHIN_DOMAIN(
        __order == memory_order::relaxed || __order == memory_order::consume || __order == memory_order::acquire ||
            __order == memory_order::seq_cst,
        "atomic_ref: memory order argument to atomic wait operation is invalid");
    std::__atomic_wait(*this, __old, __order);
  }
  _LIBCPP_HIDE_FROM_ABI void notify_one() const noexcept
    requires(!is_const_v<_Tp>)
  {
    std::__atomic_notify_one(*this);
  }
  _LIBCPP_HIDE_FROM_ABI void notify_all() const noexcept
    requires(!is_const_v<_Tp>)
  {
    std::__atomic_notify_all(*this);
  }

protected:
  using _Aligned_Tp [[__gnu__::__aligned__(required_alignment)]] = _Tp;
  _Aligned_Tp* __ptr_;

  _LIBCPP_HIDE_FROM_ABI __atomic_ref_base(_Tp& __obj) : __ptr_(std::addressof(__obj)) {}
};

template <class _Tp>
struct __atomic_waitable_traits<__atomic_ref_base<_Tp>> {
  static _LIBCPP_HIDE_FROM_ABI _Tp __atomic_load(const __atomic_ref_base<_Tp>& __a, memory_order __order) {
    return __a.load(__order);
  }
  static _LIBCPP_HIDE_FROM_ABI const _Tp* __atomic_contention_address(const __atomic_ref_base<_Tp>& __a) {
    return __a.__ptr_;
  }
};

template <class _Tp>
struct atomic_ref : public __atomic_ref_base<_Tp> {
  static_assert(is_trivially_copyable_v<_Tp>, "std::atomic_ref<T> requires that 'T' be a trivially copyable type");

  using __base = __atomic_ref_base<_Tp>;

  static_assert(__base::is_always_lock_free || !is_volatile_v<_Tp>);

  _LIBCPP_HIDE_FROM_ABI explicit atomic_ref(_Tp& __obj) : __base(__obj) {
    _LIBCPP_ASSERT_ARGUMENT_WITHIN_DOMAIN(
        reinterpret_cast<uintptr_t>(std::addressof(__obj)) % __base::required_alignment == 0,
        "atomic_ref ctor: referenced object must be aligned to required_alignment");
  }

  _LIBCPP_HIDE_FROM_ABI atomic_ref(const atomic_ref&) noexcept = default;

  _LIBCPP_HIDE_FROM_ABI __base::value_type operator=(__base::value_type __desired) const noexcept
    requires(!is_const_v<_Tp>)
  {
    return __base::operator=(__desired);
  }

  atomic_ref& operator=(const atomic_ref&) = delete;
};

template <class _Tp>
  requires(std::integral<_Tp> && !std::same_as<bool, __remove_cv_t<_Tp>>)
struct atomic_ref<_Tp> : public __atomic_ref_base<_Tp> {
  using __base = __atomic_ref_base<_Tp>;

  static_assert(__base::is_always_lock_free || !is_volatile_v<_Tp>);

  using difference_type = __base::value_type;
  using value_type      = __base::value_type;

  _LIBCPP_HIDE_FROM_ABI explicit atomic_ref(_Tp& __obj) : __base(__obj) {
    _LIBCPP_ASSERT_ARGUMENT_WITHIN_DOMAIN(
        reinterpret_cast<uintptr_t>(std::addressof(__obj)) % __base::required_alignment == 0,
        "atomic_ref ctor: referenced object must be aligned to required_alignment");
  }

  _LIBCPP_HIDE_FROM_ABI atomic_ref(const atomic_ref&) noexcept = default;

  _LIBCPP_HIDE_FROM_ABI value_type operator=(value_type __desired) const noexcept
    requires(!is_const_v<_Tp>)
  {
    return __base::operator=(__desired);
  }

  atomic_ref& operator=(const atomic_ref&) = delete;

  _LIBCPP_HIDE_FROM_ABI value_type
  fetch_add(value_type __arg, memory_order __order = memory_order_seq_cst) const noexcept
    requires(!is_const_v<value_type>)
  {
    return __atomic_fetch_add(this->__ptr_, __arg, std::__to_gcc_order(__order));
  }
  _LIBCPP_HIDE_FROM_ABI value_type
  fetch_sub(value_type __arg, memory_order __order = memory_order_seq_cst) const noexcept
    requires(!is_const_v<value_type>)
  {
    return __atomic_fetch_sub(this->__ptr_, __arg, std::__to_gcc_order(__order));
  }
  _LIBCPP_HIDE_FROM_ABI value_type
  fetch_and(value_type __arg, memory_order __order = memory_order_seq_cst) const noexcept
    requires(!is_const_v<value_type>)
  {
    return __atomic_fetch_and(this->__ptr_, __arg, std::__to_gcc_order(__order));
  }
  _LIBCPP_HIDE_FROM_ABI value_type
  fetch_or(value_type __arg, memory_order __order = memory_order_seq_cst) const noexcept
    requires(!is_const_v<value_type>)
  {
    return __atomic_fetch_or(this->__ptr_, __arg, std::__to_gcc_order(__order));
  }
  _LIBCPP_HIDE_FROM_ABI value_type
  fetch_xor(value_type __arg, memory_order __order = memory_order_seq_cst) const noexcept
    requires(!is_const_v<value_type>)
  {
    return __atomic_fetch_xor(this->__ptr_, __arg, std::__to_gcc_order(__order));
  }

  _LIBCPP_HIDE_FROM_ABI value_type operator++(int) const noexcept
    requires(!is_const_v<_Tp>)
  {
    return fetch_add(value_type(1));
  }
  _LIBCPP_HIDE_FROM_ABI value_type operator--(int) const noexcept
    requires(!is_const_v<_Tp>)
  {
    return fetch_sub(value_type(1));
  }
  _LIBCPP_HIDE_FROM_ABI value_type operator++() const noexcept
    requires(!is_const_v<_Tp>)
  {
    return fetch_add(value_type(1)) + value_type(1);
  }
  _LIBCPP_HIDE_FROM_ABI value_type operator--() const noexcept
    requires(!is_const_v<_Tp>)
  {
    return fetch_sub(value_type(1)) - value_type(1);
  }
  _LIBCPP_HIDE_FROM_ABI value_type operator+=(value_type __arg) const noexcept
    requires(!is_const_v<_Tp>)
  {
    return fetch_add(__arg) + __arg;
  }
  _LIBCPP_HIDE_FROM_ABI value_type operator-=(value_type __arg) const noexcept
    requires(!is_const_v<_Tp>)
  {
    return fetch_sub(__arg) - __arg;
  }
  _LIBCPP_HIDE_FROM_ABI value_type operator&=(value_type __arg) const noexcept
    requires(!is_const_v<_Tp>)
  {
    return fetch_and(__arg) & __arg;
  }
  _LIBCPP_HIDE_FROM_ABI value_type operator|=(value_type __arg) const noexcept
    requires(!is_const_v<_Tp>)
  {
    return fetch_or(__arg) | __arg;
  }
  _LIBCPP_HIDE_FROM_ABI value_type operator^=(value_type __arg) const noexcept
    requires(!is_const_v<_Tp>)
  {
    return fetch_xor(__arg) ^ __arg;
  }
};

template <class _Tp>
  requires std::floating_point<_Tp>
struct atomic_ref<_Tp> : public __atomic_ref_base<_Tp> {
  using __base = __atomic_ref_base<_Tp>;

  static_assert(__base::is_always_lock_free || !is_volatile_v<_Tp>);

  using difference_type = __base::value_type;
  using value_type      = __base::value_type;

  _LIBCPP_HIDE_FROM_ABI explicit atomic_ref(_Tp& __obj) : __base(__obj) {
    _LIBCPP_ASSERT_ARGUMENT_WITHIN_DOMAIN(
        reinterpret_cast<uintptr_t>(std::addressof(__obj)) % __base::required_alignment == 0,
        "atomic_ref ctor: referenced object must be aligned to required_alignment");
  }

  _LIBCPP_HIDE_FROM_ABI atomic_ref(const atomic_ref&) noexcept = default;

  _LIBCPP_HIDE_FROM_ABI value_type operator=(value_type __desired) const noexcept
    requires(!is_const_v<_Tp>)
  {
    return __base::operator=(__desired);
  }

  atomic_ref& operator=(const atomic_ref&) = delete;

  _LIBCPP_HIDE_FROM_ABI value_type
  fetch_add(value_type __arg, memory_order __order = memory_order_seq_cst) const noexcept
    requires(!is_const_v<_Tp>)
  {
    value_type __old = this->load(memory_order_relaxed);
    value_type __new = __old + __arg;
    while (!this->compare_exchange_weak(__old, __new, __order, memory_order_relaxed)) {
      __new = __old + __arg;
    }
    return __old;
  }
  _LIBCPP_HIDE_FROM_ABI value_type
  fetch_sub(value_type __arg, memory_order __order = memory_order_seq_cst) const noexcept
    requires(!is_const_v<_Tp>)
  {
    value_type __old = this->load(memory_order_relaxed);
    value_type __new = __old - __arg;
    while (!this->compare_exchange_weak(__old, __new, __order, memory_order_relaxed)) {
      __new = __old - __arg;
    }
    return __old;
  }

  _LIBCPP_HIDE_FROM_ABI value_type operator+=(value_type __arg) const noexcept
    requires(!is_const_v<_Tp>)
  {
    return fetch_add(__arg) + __arg;
  }
  _LIBCPP_HIDE_FROM_ABI value_type operator-=(value_type __arg) const noexcept
    requires(!is_const_v<_Tp>)
  {
    return fetch_sub(__arg) - __arg;
  }
};

template <class _Tp>
  requires(std::is_pointer_v<_Tp>)
struct atomic_ref<_Tp> : public __atomic_ref_base<_Tp> {
  using __base = __atomic_ref_base<_Tp>;

  using difference_type = ptrdiff_t;
  using value_type      = typename __base::value_type;

  _LIBCPP_HIDE_FROM_ABI explicit atomic_ref(_Tp& __ptr) : __base(__ptr) {}

  _LIBCPP_HIDE_FROM_ABI value_type operator=(value_type __desired) const noexcept
    requires(!is_const_v<_Tp>)
  {
    return __base::operator=(__desired);
  }

  atomic_ref& operator=(const atomic_ref&) = delete;

  _LIBCPP_HIDE_FROM_ABI value_type
  fetch_add(ptrdiff_t __arg, memory_order __order = memory_order_seq_cst) const noexcept
    requires(!is_const_v<_Tp>)
  {
    return __atomic_fetch_add(
        this->__ptr_, __arg * sizeof(__remove_pointer_t<value_type>), std::__to_gcc_order(__order));
  }
  _LIBCPP_HIDE_FROM_ABI value_type
  fetch_sub(ptrdiff_t __arg, memory_order __order = memory_order_seq_cst) const noexcept
    requires(!is_const_v<_Tp>)
  {
    return __atomic_fetch_sub(
        this->__ptr_, __arg * sizeof(__remove_pointer_t<value_type>), std::__to_gcc_order(__order));
  }

  _LIBCPP_HIDE_FROM_ABI value_type operator++(int) const noexcept
    requires(!is_const_v<_Tp>)
  {
    return fetch_add(1);
  }
  _LIBCPP_HIDE_FROM_ABI value_type operator--(int) const noexcept
    requires(!is_const_v<_Tp>)
  {
    return fetch_sub(1);
  }
  _LIBCPP_HIDE_FROM_ABI value_type operator++() const noexcept
    requires(!is_const_v<_Tp>)
  {
    return fetch_add(1) + 1;
  }
  _LIBCPP_HIDE_FROM_ABI value_type operator--() const noexcept
    requires(!is_const_v<_Tp>)
  {
    return fetch_sub(1) - 1;
  }
  _LIBCPP_HIDE_FROM_ABI value_type operator+=(ptrdiff_t __arg) const noexcept
    requires(!is_const_v<_Tp>)
  {
    return fetch_add(__arg) + __arg;
  }
  _LIBCPP_HIDE_FROM_ABI value_type operator-=(ptrdiff_t __arg) const noexcept
    requires(!is_const_v<_Tp>)
  {
    return fetch_sub(__arg) - __arg;
  }
};

_LIBCPP_CTAD_SUPPORTED_FOR_TYPE(atomic_ref);

#endif // _LIBCPP_STD_VER >= 20

_LIBCPP_END_NAMESPACE_STD

_LIBCPP_POP_MACROS

#endif // _LIBCPP__ATOMIC_ATOMIC_REF_H
