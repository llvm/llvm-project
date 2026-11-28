//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___VECTOR_VECTOR2_SWAP_H
#define _LIBCPP___VECTOR_VECTOR2_SWAP_H

#include <__config>
#include <__fwd/vector2.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

#if _LIBCPP_STD_VER >= 29

_LIBCPP_BEGIN_NAMESPACE_STD

template <class _Tp, class _Allocator>
_LIBCPP_CONSTEXPR_SINCE_CXX20 inline _LIBCPP_HIDE_FROM_ABI void
swap(vector2<_Tp, _Allocator>& __x, vector2<_Tp, _Allocator>& __y) _NOEXCEPT_(_NOEXCEPT_(__x.swap(__y))) {
  __x.swap(__y);
}

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP_STD_VER >= 29

#endif // _LIBCPP___VECTOR_VECTOR2_SWAP_H
