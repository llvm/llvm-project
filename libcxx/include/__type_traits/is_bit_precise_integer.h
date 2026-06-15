//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___TYPE_TRAITS_IS_BIT_PRECISE_INTEGER_H
#define _LIBCPP___TYPE_TRAITS_IS_BIT_PRECISE_INTEGER_H

#include <__config>
#include <__type_traits/remove_cv.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

// True iff T (cv stripped) is _BitInt(N) or unsigned _BitInt(N).

template <class _Tp>
inline const bool __is_bit_precise_integer_impl_v = false;

#ifdef __BITINT_MAXWIDTH__
template <int _Nb>
inline const bool __is_bit_precise_integer_impl_v<_BitInt(_Nb)> = true;
template <int _Nb>
inline const bool __is_bit_precise_integer_impl_v<unsigned _BitInt(_Nb)> = true;
#endif

template <class _Tp>
inline const bool __is_bit_precise_integer_v = __is_bit_precise_integer_impl_v<__remove_cv_t<_Tp> >;

// Gate predicate: standard integers always admit; bit-precise integers admit
// only with _LIBCPP_HAS_BITINT_EXTENSIONS on.
template <class _Tp>
inline const bool __admits_bitint_extension_v = _LIBCPP_HAS_BITINT_EXTENSIONS || !__is_bit_precise_integer_v<_Tp>;

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___TYPE_TRAITS_IS_BIT_PRECISE_INTEGER_H
