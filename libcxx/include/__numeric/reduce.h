// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___NUMERIC_REDUCE_H
#define _LIBCPP___NUMERIC_REDUCE_H

#include <__config>
#include <__functional/operations.h>
#include <__fwd/complex.h>
#include <__iterator/iterator_traits.h>
#include <__type_traits/desugars_to.h>
#include <__type_traits/is_floating_point.h>
#include <__type_traits/is_integral.h>
#include <__type_traits/is_same.h>
#include <__type_traits/is_specialization.h>
#include <__utility/move.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_PUSH_MACROS
#include <__undef_macros>

#if _LIBCPP_STD_VER >= 17

_LIBCPP_BEGIN_NAMESPACE_STD

// If the optimization is geared towards reducing the binary size, do not use tree-based reduction unrolling.
// Otherwise, allow it if the iterator is random-access.
template <class _InputIterator>
inline constexpr bool __is_reduction_tree_enabled_v =
#  if defined(__OPTIMIZE_SIZE__)
    false;
#  else
    __has_random_access_iterator_category<_InputIterator>::value;
#  endif

// Try to deduce upfront whether the compiler can efficiently vectorize a reduction loop given the iterator type, the
// value type and the reduction operation.
// The logic is based on these heuristics:
// - The vectorization pragma must be available
// - The iterator type must be contiguous
// - The operand types must be either integral, floating-point or std::complex
// - The operation must be std::plus
template <class _InputIterator, class _Tp, class _BinaryOp>
inline constexpr bool __is_reduction_loop_vectorizable_v =
#  if _LIBCPP_HAS_LOOP_VECTORIZATION_PRAGMA
    __libcpp_is_contiguous_iterator<_InputIterator>::value && is_same_v<_Tp, __iterator_value_type<_InputIterator>> &&
    (is_integral_v<_Tp> || is_floating_point_v<_Tp> || __is_specialization_v<_Tp, complex>) &&
    __desugars_to_v<std::__plus_tag, _BinaryOp, _Tp, __iterator_value_type<_InputIterator>>;
#  else
    false;
#  endif

template <class _InputIterator, class _Tp, class _BinaryOp>
[[nodiscard]] _LIBCPP_HIDE_FROM_ABI
_LIBCPP_CONSTEXPR_SINCE_CXX20 _Tp reduce(_InputIterator __first, _InputIterator __last, _Tp __init, _BinaryOp __b) {
  if constexpr (__is_reduction_tree_enabled_v<_InputIterator> &&
                !__is_reduction_loop_vectorizable_v<_InputIterator, _Tp, _BinaryOp>) {
    // Unroll the loop using a 4-way tree reduction. This is more efficient than processing element by element.
    // However, when the compiler can vectorize the reduction, the tree-based unrolling can be slower, so only do it
    // when vectorization is not expected.
    for (; __last - __first >= 4; __first += 4)
      __init = __b(std::move(__init), __b(__b(__first[0], __first[1]), __b(__first[2], __first[3])));
  }
  for (; __first != __last; ++__first)
    __init = __b(std::move(__init), *__first);
  return __init;
}

template <class _InputIterator, class _Tp>
[[nodiscard]] _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX20 _Tp
reduce(_InputIterator __first, _InputIterator __last, _Tp __init) {
  return std::reduce(__first, __last, __init, std::plus<>());
}

template <class _InputIterator>
[[nodiscard]] _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX20 typename iterator_traits<_InputIterator>::value_type
reduce(_InputIterator __first, _InputIterator __last) {
  return std::reduce(__first, __last, typename iterator_traits<_InputIterator>::value_type{});
}

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP_STD_VER >= 17

_LIBCPP_POP_MACROS

#endif // _LIBCPP___NUMERIC_REDUCE_H
