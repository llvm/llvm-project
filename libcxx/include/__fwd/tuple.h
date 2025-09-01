//===---------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//

#ifndef _LIBCPP___FWD_TUPLE_H
#define _LIBCPP___FWD_TUPLE_H

#include <__config>
#include <__cstddef/size_t.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

template <size_t, class>
struct tuple_element;

#ifndef _LIBCPP_CXX03_LANG

template <class...>
class tuple;

template <class>
inline const bool __is_tuple_v = false;

template <class... _Tp>
inline const bool __is_tuple_v<tuple<_Tp...>> = true;

template <size_t _Ip, class... _Tp>
struct tuple_element<_Ip, tuple<_Tp...> > {
  using type _LIBCPP_NODEBUG = __type_pack_element<_Ip, _Tp...>;
};

template <class>
struct tuple_size;

template <size_t _Ip, class... _Tp>
_LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX14 typename tuple_element<_Ip, tuple<_Tp...> >::type&
get(tuple<_Tp...>&) _NOEXCEPT;

template <size_t _Ip, class... _Tp>
_LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX14 const typename tuple_element<_Ip, tuple<_Tp...> >::type&
get(const tuple<_Tp...>&) _NOEXCEPT;

template <size_t _Ip, class... _Tp>
_LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX14 typename tuple_element<_Ip, tuple<_Tp...> >::type&&
get(tuple<_Tp...>&&) _NOEXCEPT;

template <size_t _Ip, class... _Tp>
_LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX14 const typename tuple_element<_Ip, tuple<_Tp...> >::type&&
get(const tuple<_Tp...>&&) _NOEXCEPT;

#endif // _LIBCPP_CXX03_LANG

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___FWD_TUPLE_H
