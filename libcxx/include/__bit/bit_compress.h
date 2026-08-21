//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___BIT_BIT_COMPRESS_H
#define _LIBCPP___BIT_BIT_COMPRESS_H

#include <__config>
#include <__type_traits/integer_traits.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

#if _LIBCPP_STD_VER >= 29

_LIBCPP_BEGIN_NAMESPACE_STD

template <__unsigned_integer _Tp>
_LIBCPP_HIDE_FROM_ABI constexpr _Tp bit_compress(_Tp __t, _Tp __m) noexcept {
  return __builtin_elementwise_pext(__t, __m);
}

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP_STD_VER >= 29

#endif // _LIBCPP___BIT_BIT_COMPRESS_H
