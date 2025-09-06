//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___MEMORY_RELOCATE_AT_H
#define _LIBCPP___MEMORY_RELOCATE_AT_H

#include <__memory/allocator_traits.h>
#include <__memory/construct_at.h>
#include <__type_traits/enable_if.h>
#include <__type_traits/integral_constant.h>
#include <__type_traits/is_constant_evaluated.h>
#include <__type_traits/is_constructible.h>
#include <__type_traits/is_nothrow_relocatable.h>
#include <__type_traits/is_same.h>
#include <__type_traits/is_trivially_relocatable.h>
#include <__utility/move.h>
#include <__utility/scope_guard.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_PUSH_MACROS
#include <__undef_macros>

_LIBCPP_BEGIN_NAMESPACE_STD

template <class _Source, class _Dest>
struct __relocatable_from
    : _BoolConstant<_IsSame<_Source, _Dest>::value &&
                    (is_move_constructible<_Dest>::value || __libcpp_is_trivially_relocatable<_Dest>::value)> {};

template <class _Tp>
struct __destroy_object {
  _LIBCPP_CONSTEXPR_SINCE_CXX20 _LIBCPP_HIDE_FROM_ABI __destroy_object(_Tp* __obj) : __obj_(__obj) {}
  _LIBCPP_CONSTEXPR_SINCE_CXX20 _LIBCPP_HIDE_FROM_ABI void operator()() const { std::__destroy_at(__obj_); }
  _Tp* __obj_;
};

template <class _Tp>
_LIBCPP_HIDE_FROM_ABI _Tp* __libcpp_builtin_trivially_relocate_at(_Tp* __source, _Tp* __dest) _NOEXCEPT {
  static_assert(__libcpp_is_trivially_relocatable<_Tp>::value, "");
  // Casting to void* to suppress clang complaining that this is technically UB.
  __builtin_memcpy(static_cast<void*>(__dest), __source, sizeof(_Tp));
  return __dest;
}

template <class _Tp>
_LIBCPP_HIDE_FROM_ABI _Tp* __libcpp_builtin_trivially_relocate_at(_Tp* __first, _Tp* __last, _Tp* __dest) _NOEXCEPT {
  static_assert(__libcpp_is_trivially_relocatable<_Tp>::value, "");
  // Casting to void* to suppress clang complaining that this is technically UB.
  __builtin_memmove(static_cast<void*>(__dest), __first, (__last - __first) * sizeof(_Tp));
  return __dest;
}

template <class _Tp,
          __enable_if_t<__relocatable_from<_Tp, _Tp>::value, int>                                                = 0,
          __enable_if_t<__libcpp_is_trivially_relocatable<_Tp>::value && is_move_constructible<_Tp>::value, int> = 0>
_LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX20 _Tp* __relocate_at(_Tp* __dest, _Tp* __source)
    _NOEXCEPT_(__is_nothrow_relocatable<_Tp>::value) {
  if (__libcpp_is_constant_evaluated()) {
    auto __guard = std::__make_scope_guard(__destroy_object<_Tp>(__source));
    return std::__construct_at(__dest, std::move(*__source));
  } else {
    return std::__libcpp_builtin_trivially_relocate_at(__source, __dest);
  }
}

template <class _Tp,
          __enable_if_t<__relocatable_from<_Tp, _Tp>::value, int>                                                 = 0,
          __enable_if_t<__libcpp_is_trivially_relocatable<_Tp>::value && !is_move_constructible<_Tp>::value, int> = 0>
_LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX20 _Tp* __relocate_at(_Tp* __dest, _Tp* __source)
    _NOEXCEPT_(__is_nothrow_relocatable<_Tp>::value) {
  return std::__libcpp_builtin_trivially_relocate_at(__source, __dest);
}

template <class _Tp,
          __enable_if_t<__relocatable_from<_Tp, _Tp>::value, int>            = 0,
          __enable_if_t<!__libcpp_is_trivially_relocatable<_Tp>::value, int> = 0>
_LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX20 _Tp* __relocate_at(_Tp* __dest, _Tp* __source)
    _NOEXCEPT_(__is_nothrow_relocatable<_Tp>::value) {
  auto __guard = std::__make_scope_guard(__destroy_object<_Tp>(__source));
  return std::__construct_at(__dest, std::move(*__source));
}

_LIBCPP_END_NAMESPACE_STD

_LIBCPP_POP_MACROS

#endif // _LIBCPP___MEMORY_RELOCATE_AT_H
