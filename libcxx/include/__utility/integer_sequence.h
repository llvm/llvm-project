//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___UTILITY_INTEGER_SEQUENCE_H
#define _LIBCPP___UTILITY_INTEGER_SEQUENCE_H

#include <__config>
#include <__cstddef/size_t.h>
#include <__type_traits/is_integral.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

#ifndef _LIBCPP_CXX03_LANG

_LIBCPP_BEGIN_NAMESPACE_STD

#  if __has_builtin(__make_integer_seq)
template <template <class _Tp, _Tp...> class _BaseType, class _Tp, _Tp _SequenceSize>
using __make_integer_sequence_impl _LIBCPP_NODEBUG = __make_integer_seq<_BaseType, _Tp, _SequenceSize>;
#  else
template <template <class _Tp, _Tp...> class _BaseType, class _Tp, _Tp _SequenceSize>
using __make_integer_sequence_impl _LIBCPP_NODEBUG = _BaseType<_Tp, __integer_pack(_SequenceSize)...>;
#  endif

template <class _Tp, _Tp... _Indices>
struct __integer_sequence {
  using value_type = _Tp;
  static_assert(is_integral<_Tp>::value, "std::integer_sequence can only be instantiated with an integral type");
  static _LIBCPP_HIDE_FROM_ABI constexpr size_t size() noexcept { return sizeof...(_Indices); }
};

template <size_t... _Indices>
using __index_sequence _LIBCPP_NODEBUG = __integer_sequence<size_t, _Indices...>;

template <size_t _SequenceSize>
using __make_index_sequence _LIBCPP_NODEBUG = __make_integer_sequence_impl<__integer_sequence, size_t, _SequenceSize>;

#  if _LIBCPP_STD_VER >= 14

template <class _Tp, _Tp... _Indices>
struct integer_sequence : __integer_sequence<_Tp, _Indices...> {};

template <size_t... _Ip>
using index_sequence = integer_sequence<size_t, _Ip...>;

template <class _Tp, _Tp _Ep>
using make_integer_sequence _LIBCPP_NODEBUG = __make_integer_sequence_impl<integer_sequence, _Tp, _Ep>;

template <size_t _Np>
using make_index_sequence = make_integer_sequence<size_t, _Np>;

template <class... _Tp>
using index_sequence_for = make_index_sequence<sizeof...(_Tp)>;

#    if _LIBCPP_STD_VER >= 20
// Executes __func for every element in an index_sequence.
template <size_t... _Index, class _Function>
_LIBCPP_HIDE_FROM_ABI constexpr void __for_each_index_sequence(index_sequence<_Index...>, _Function __func) {
  (__func.template operator()<_Index>(), ...);
}
#    endif // _LIBCPP_STD_VER >= 20

#  endif // _LIBCPP_STD_VER >= 14

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP_CXX03_LANG

#endif // _LIBCPP___UTILITY_INTEGER_SEQUENCE_H
