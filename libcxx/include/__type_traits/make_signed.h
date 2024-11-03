//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___TYPE_TRAITS_MAKE_SIGNED_H
#define _LIBCPP___TYPE_TRAITS_MAKE_SIGNED_H

#include <__config>
#include <__type_traits/copy_cv.h>
#include <__type_traits/remove_cv.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

#if __has_builtin(__make_signed)

template <class _Tp>
using __make_signed_t = __make_signed(_Tp);

#else
template <class _Tp>
struct __make_signed{};

// clang-format off
template <> struct __make_signed<bool              > {};
template <> struct __make_signed<  signed short    > {typedef short     type;};
template <> struct __make_signed<unsigned short    > {typedef short     type;};
template <> struct __make_signed<  signed int      > {typedef int       type;};
template <> struct __make_signed<unsigned int      > {typedef int       type;};
template <> struct __make_signed<  signed long     > {typedef long      type;};
template <> struct __make_signed<unsigned long     > {typedef long      type;};
template <> struct __make_signed<  signed long long> {typedef long long type;};
template <> struct __make_signed<unsigned long long> {typedef long long type;};
#  if _LIBCPP_HAS_INT128
template <> struct __make_signed<__int128_t        > {typedef __int128_t type;};
template <> struct __make_signed<__uint128_t       > {typedef __int128_t type;};
#  endif
// clang-format on

template <class _Tp>
using __make_signed_t = __copy_cv_t<_Tp, typename __make_signed<__remove_cv_t<_Tp> >::type>;

#endif // __has_builtin(__make_signed)

template <class _Tp>
struct make_signed {
  using type _LIBCPP_NODEBUG = __make_signed_t<_Tp>;
};

#if _LIBCPP_STD_VER >= 14
template <class _Tp>
using make_signed_t = __make_signed_t<_Tp>;
#endif

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___TYPE_TRAITS_MAKE_SIGNED_H
