//===---------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//

#ifndef _LIBCPP___FWD_PAIR_H
#define _LIBCPP___FWD_PAIR_H

#include <__config>
#include <__cstddef/size_t.h>
#include <__fwd/tuple.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

template <class, class>
struct pair;

template <class _Type>
inline const bool __is_pair_v = false;

template <class _Type1, class _Type2>
inline const bool __is_pair_v<pair<_Type1, _Type2> > = true;

// clang-format fails miserably with pack subscripting currently
// clang-format off
template <size_t _Ip, class... _Ts>
_LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX14 _Ts...[_Ip]& get(pair<_Ts...>&) _NOEXCEPT;

template <size_t _Ip, class... _Ts>
_LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX14 const _Ts...[_Ip]& get(const pair<_Ts...>&) _NOEXCEPT;

#ifndef _LIBCPP_CXX03_LANG
template <size_t _Ip, class... _Ts>
_LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX14 _Ts...[_Ip]&& get(pair<_Ts...>&&) _NOEXCEPT;

template <size_t _Ip, class... _Ts>
_LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX14 const _Ts...[_Ip]&& get(const pair<_Ts...>&&) _NOEXCEPT;
#endif
// clang-format on

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___FWD_PAIR_H
