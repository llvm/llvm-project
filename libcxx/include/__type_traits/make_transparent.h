//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___TYPE_TRAITS_MAKE_TRANSPARENT_H
#define _LIBCPP___TYPE_TRAITS_MAKE_TRANSPARENT_H

#include <__config>
#include <__type_traits/enable_if.h>
#include <__type_traits/is_empty.h>
#include <__type_traits/is_same.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

// __make_transparent tries to create a transparent comparator from its non-transparent counterpart, e.g. obtain
// `less<>` from `less<T>`. This is useful in cases where conversions can be avoided (e.g. a string literal to a
// std::string). This depends on the argument type provided to the comparator, because a comparator might be
// transparent for some argument types but not for others.

template <class _ArgumentType, class _Comparator>
struct __make_transparent {
  using type _LIBCPP_NODEBUG = _Comparator;
};

template <class _ArgumentType, class _Comparator>
using __make_transparent_t _LIBCPP_NODEBUG = typename __make_transparent<_ArgumentType, _Comparator>::type;

template <class _ArgumentType,
          class _Comparator,
          __enable_if_t<is_same<_Comparator, __make_transparent_t<_ArgumentType, _Comparator> >::value, int> = 0>
_LIBCPP_HIDE_FROM_ABI _Comparator& __as_transparent(_Comparator& __comp) {
  return __comp;
}

template <class _ArgumentType,
          class _Comparator,
          __enable_if_t<!is_same<_Comparator, __make_transparent_t<_ArgumentType, _Comparator> >::value, int> = 0>
_LIBCPP_HIDE_FROM_ABI __make_transparent_t<_ArgumentType, _Comparator> __as_transparent(_Comparator&) {
  static_assert(is_empty<_Comparator>::value);
  return __make_transparent_t<_ArgumentType, _Comparator>();
}

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___TYPE_TRAITS_MAKE_TRANSPARENT_H
