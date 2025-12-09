//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___TYPE_TRAITS_ALIGNED_STORAGE_H
#define _LIBCPP___TYPE_TRAITS_ALIGNED_STORAGE_H

#include <__config>
#include <__cstddef/size_t.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

template <class _Tp>
struct _ALIGNAS(_LIBCPP_PREFERRED_ALIGNOF(_Tp)) _AlignedAsT {};

template <class... _Args>
struct __max_align_impl : _AlignedAsT<_Args>... {};

struct __struct_double {
  long double __lx;
};
struct __struct_double4 {
  double __lx[4];
};

inline const size_t __aligned_storage_max_align =
    _LIBCPP_ALIGNOF(__max_align_impl<unsigned long long, double, long double, __struct_double, __struct_double4, int*>);

template <size_t _Len>
inline const size_t __aligned_storage_alignment =
    _Len > __aligned_storage_max_align
        ? __aligned_storage_max_align
        : size_t(1) << ((sizeof(size_t) * __CHAR_BIT__) - __builtin_clzg(_Len) - 1);

template <size_t _Len, size_t _Align = __aligned_storage_alignment<_Len> >
struct _LIBCPP_DEPRECATED_IN_CXX23 _LIBCPP_NO_SPECIALIZATIONS aligned_storage {
  union _ALIGNAS(_Align) type {
    unsigned char __data[(_Len + _Align - 1) / _Align * _Align];
  };
};

#if _LIBCPP_STD_VER >= 14

_LIBCPP_SUPPRESS_DEPRECATED_PUSH
template <size_t _Len, size_t _Align = __aligned_storage_alignment<_Len> >
using aligned_storage_t _LIBCPP_DEPRECATED_IN_CXX23 = typename aligned_storage<_Len, _Align>::type;
_LIBCPP_SUPPRESS_DEPRECATED_POP

#endif

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___TYPE_TRAITS_ALIGNED_STORAGE_H
