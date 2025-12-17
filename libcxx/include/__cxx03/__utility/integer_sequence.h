//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___CXX03___UTILITY_INTEGER_SEQUENCE_H
#define _LIBCPP___CXX03___UTILITY_INTEGER_SEQUENCE_H

#include <__cxx03/__config>
#include <__cxx03/__type_traits/is_integral.h>
#include <__cxx03/cstddef>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

template <size_t...>
struct __tuple_indices;

template <class _IdxType, _IdxType... _Values>
struct __integer_sequence {
  template <template <class _OIdxType, _OIdxType...> class _ToIndexSeq, class _ToIndexType>
  using __convert = _ToIndexSeq<_ToIndexType, _Values...>;

  template <size_t _Sp>
  using __to_tuple_indices = __tuple_indices<(_Values + _Sp)...>;
};

#if __has_builtin(__make_integer_seq)
template <size_t _Ep, size_t _Sp>
using __make_indices_imp =
    typename __make_integer_seq<__integer_sequence, size_t, _Ep - _Sp>::template __to_tuple_indices<_Sp>;
#elif __has_builtin(__integer_pack)
template <size_t _Ep, size_t _Sp>
using __make_indices_imp =
    typename __integer_sequence<size_t, __integer_pack(_Ep - _Sp)...>::template __to_tuple_indices<_Sp>;
#else
#  error "No known way to get an integer pack from the compiler"
#endif

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___CXX03___UTILITY_INTEGER_SEQUENCE_H
