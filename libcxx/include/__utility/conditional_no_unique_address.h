//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___UTILITY_CONDITIONAL_NO_UNIQUE_ADDRESS_H
#define _LIBCPP___UTILITY_CONDITIONAL_NO_UNIQUE_ADDRESS_H

#include <__config>
#include <__type_traits/invoke.h>
#include <__type_traits/is_swappable.h>
#include <__utility/forward.h>
#include <__utility/in_place.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_PUSH_MACROS
#include <__undef_macros>

_LIBCPP_BEGIN_NAMESPACE_STD

// If parameter type `_Tp` of `__conditional_no_unique_address` is neither
// copyable nor movable, a constructor with this tag is provided. For that
// constructor, the user has to provide a function and arguments. The function
// must return an object of type `_Tp`. When the function is invoked by the
// constructor, guaranteed copy elision kicks in and the `_Tp` is constructed
// in place.
struct __conditional_no_unique_address_invoke_tag {};

// This class implements an object with `[[no_unique_address]]` conditionally applied to it,
// based on the value of `_NoUnique`.
//
// A member of this class must always have `[[no_unique_address]]` applied to
// it. Otherwise, the `[[no_unique_address]]` in the "`_NoUnique == true`" case
// would not have any effect. In the `false` case, the `__v` is not
// `[[no_unique_address]]`, so nullifies the effects of the "outer"
// `[[no_unique_address]]` regarding data layout.
//
// If we had a language feature, this class would basically be replaced by `[[no_unique_address(condition)]]`.
template <bool _NoUnique, class _Tp>
struct __conditional_no_unique_address;

template <class _Tp>
struct __conditional_no_unique_address<true, _Tp> {
  template <class... _Args>
  _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR explicit __conditional_no_unique_address(__in_place_t, _Args&&... __args)
      : __v(std::forward<_Args>(__args)...) {}

  template <class _Func, class... _Args>
  _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR explicit __conditional_no_unique_address(
      __conditional_no_unique_address_invoke_tag, _Func&& __f, _Args&&... __args)
      : __v(std::__invoke(std::forward<_Func>(__f), std::forward<_Args>(__args)...)) {}

  _LIBCPP_NO_UNIQUE_ADDRESS _Tp __v;
};

template <class _Tp>
struct __conditional_no_unique_address<false, _Tp> {
  template <class... _Args>
  _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR explicit __conditional_no_unique_address(__in_place_t, _Args&&... __args)
      : __v(std::forward<_Args>(__args)...) {}

  template <class _Func, class... _Args>
  _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR explicit __conditional_no_unique_address(
      __conditional_no_unique_address_invoke_tag, _Func&& __f, _Args&&... __args)
      : __v(std::__invoke(std::forward<_Func>(__f), std::forward<_Args>(__args)...)) {}

  _Tp __v;
};

template <bool _NoUnique, class _Tp>
void swap(__conditional_no_unique_address<_NoUnique, _Tp>& __lhs,
          __conditional_no_unique_address<_NoUnique, _Tp>& __rhs) _NOEXCEPT_(__is_swappable_v<_Tp>) {
  using std::swap;
  swap(__lhs.__v, __rhs.__v);
}

_LIBCPP_END_NAMESPACE_STD

_LIBCPP_POP_MACROS

#endif // _LIBCPP___UTILITY_CONDITIONAL_NO_UNIQUE_ADDRESS_H
