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
#include <__iterator/iterator_traits.h>
#include <__type_traits/is_integral.h>
#include <__type_traits/is_trivially_copyable.h>
#include <__utility/move.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_PUSH_MACROS
#include <__undef_macros>

#if _LIBCPP_STD_VER >= 17

_LIBCPP_BEGIN_NAMESPACE_STD

template <class _InputIterator, class _Tp, class _BinaryOp>
[[nodiscard]] _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX20 _Tp
reduce(_InputIterator __first, _InputIterator __last, _Tp __init, _BinaryOp __b) {
#  ifndef __OPTIMIZE_SIZE__
  if constexpr (__has_random_access_iterator_category<_InputIterator>::value && //
                is_trivially_copyable<_Tp>::value &&                            //
                !is_integral_v<_Tp>) {
    // Unroll the loop using a 4-way tree reduction.
    // This is often more efficient than processing element by element.
    // However, conservatively: apply it only for trivial types that are also not integral
    // (presumably the compiler is free to optimize operations with integral types really well).
    for (; __last - __first >= 4; __first += 4)
      __init = __b(std::move(__init), __b(__b(__first[0], __first[1]), __b(__first[2], __first[3])));
  }
#  endif
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
