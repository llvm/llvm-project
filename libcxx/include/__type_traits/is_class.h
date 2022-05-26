//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___TYPE_TRAITS_IS_CLASS_H
#define _LIBCPP___TYPE_TRAITS_IS_CLASS_H

#include <__config>
#include <__type_traits/integral_constant.h>
#include <__type_traits/is_union.h>
#include <__type_traits/remove_cv.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

#if __has_feature(is_class) || defined(_LIBCPP_COMPILER_GCC)

template <class _Tp> struct _LIBCPP_TEMPLATE_VIS is_class
    : public integral_constant<bool, __is_class(_Tp)> {};

#else

namespace __is_class_imp
{
template <class _Tp> char  __test(int _Tp::*);
template <class _Tp> __two __test(...);
}

template <class _Tp> struct _LIBCPP_TEMPLATE_VIS is_class
    : public integral_constant<bool, sizeof(__is_class_imp::__test<_Tp>(0)) == 1 && !is_union<_Tp>::value> {};

#endif

#if _LIBCPP_STD_VER > 14
template <class _Tp>
inline constexpr bool is_class_v = is_class<_Tp>::value;
#endif

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___TYPE_TRAITS_IS_CLASS_H
