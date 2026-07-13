//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___TYPE_TRAITS_IS_FLOATING_POINT_H
#define _LIBCPP___TYPE_TRAITS_IS_FLOATING_POINT_H

#include <__config>
#include <__type_traits/integral_constant.h>
#include <__type_traits/remove_cv.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

// clang-format off
template <class _Tp> inline const bool __is_floating_point_impl              = false;
template <>          inline const bool __is_floating_point_impl<float>       = true;
template <>          inline const bool __is_floating_point_impl<double>      = true;
template <>          inline const bool __is_floating_point_impl<long double> = true;
// clang-format on

template <class _Tp>
struct _LIBCPP_NO_SPECIALIZATIONS is_floating_point
    : integral_constant<bool, __is_floating_point_impl<__remove_cv_t<_Tp> > > {};

#if _LIBCPP_STD_VER >= 17
template <class _Tp>
_LIBCPP_NO_SPECIALIZATIONS inline constexpr bool is_floating_point_v = __is_floating_point_impl<__remove_cv_t<_Tp>>;
#endif

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___TYPE_TRAITS_IS_FLOATING_POINT_H
