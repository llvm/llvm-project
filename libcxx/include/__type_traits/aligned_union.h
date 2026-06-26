//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___TYPE_TRAITS_ALIGNED_UNION_H
#define _LIBCPP___TYPE_TRAITS_ALIGNED_UNION_H

#include <__config>
#include <__cstddef/size_t.h>
#include <__type_traits/aligned_storage.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

template <class _Arg0, class... _Args>
union __union {
  _ALIGNAS(_LIBCPP_PREFERRED_ALIGNOF(_Arg0)) _Arg0 __arg;
  __union<_Args...> __u;
};

template <class _Arg>
union __union<_Arg> {
  _ALIGNAS(_LIBCPP_PREFERRED_ALIGNOF(_Arg)) _Arg __arg;
};

template <size_t _Len, class _Type0, class... _Types>
struct _LIBCPP_DEPRECATED_IN_CXX23 _LIBCPP_NO_SPECIALIZATIONS aligned_union {
  using _Union _LIBCPP_NODEBUG        = __union<char[_Len], _Type0, _Types...>;
  static const size_t alignment_value = _LIBCPP_ALIGNOF(_Union);
  using type _LIBCPP_NODEBUG          = typename aligned_storage<sizeof(_Union), alignment_value>::type;
};

#if _LIBCPP_STD_VER >= 14
template <size_t _Len, class... _Types>
using aligned_union_t _LIBCPP_DEPRECATED_IN_CXX23 = typename aligned_union<_Len, _Types...>::type;
#endif

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___TYPE_TRAITS_ALIGNED_UNION_H
