//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___TYPE_TRAITS_IS_POLYMORPHIC_H
#define _LIBCPP___TYPE_TRAITS_IS_POLYMORPHIC_H

#include <__config>
#include <__type_traits/integral_constant.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

#if __has_feature(is_polymorphic) || defined(_LIBCPP_COMPILER_MSVC)

template <class _Tp>
struct _LIBCPP_TEMPLATE_VIS is_polymorphic
    : public integral_constant<bool, __is_polymorphic(_Tp)> {};

#else

template<typename _Tp> char &__is_polymorphic_impl(
    typename enable_if<sizeof((_Tp*)dynamic_cast<const volatile void*>(declval<_Tp*>())) != 0,
                       int>::type);
template<typename _Tp> __two &__is_polymorphic_impl(...);

template <class _Tp> struct _LIBCPP_TEMPLATE_VIS is_polymorphic
    : public integral_constant<bool, sizeof(__is_polymorphic_impl<_Tp>(0)) == 1> {};

#endif // __has_feature(is_polymorphic)

#if _LIBCPP_STD_VER > 14
template <class _Tp>
inline constexpr bool is_polymorphic_v = is_polymorphic<_Tp>::value;
#endif

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___TYPE_TRAITS_IS_POLYMORPHIC_H
