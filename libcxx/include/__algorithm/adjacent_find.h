// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___ALGORITHM_ADJACENT_FIND_H
#define _LIBCPP___ALGORITHM_ADJACENT_FIND_H

#include <__algorithm/comp.h>
#include <__algorithm/iterator_operations.h>
#include <__algorithm/simd_utils.h>
#include <__algorithm/unwrap_iter.h>
#include <__config>
#include <__iterator/iterator_traits.h>
#include <__type_traits/desugars_to.h>
#include <__type_traits/is_constant_evaluated.h>
#include <__utility/move.h>
#include <cstddef>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_PUSH_MACROS
#include <__undef_macros>

_LIBCPP_BEGIN_NAMESPACE_STD

template <class _Iter, class _Sent, class _BinaryPredicate>
[[__nodiscard__]] _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX20 _Iter
__adjacent_find_loop(_Iter __first, _Sent __last, _BinaryPredicate&& __pred) {
  if (__first == __last)
    return __first;
  _Iter __i = __first;
  while (++__i != __last) {
    if (__pred(*__first, *__i))
      return __first;
    __first = __i;
  }
  return __i;
}

template <class _Iter, class _Sent, class _BinaryPredicate>
[[__nodiscard__]] _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX20 _Iter
__adjacent_find(_Iter __first, _Sent __last, _BinaryPredicate&& __pred) {
  return std::__adjacent_find_loop(__first, __last, __pred);
}

#if _LIBCPP_VECTORIZE_ALGORITHMS

template <class _Tp, class _Pred>
[[__nodiscard__]] _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX20 _Tp*
__adjacent_find_vectorized(_Tp* __first, _Tp* __last, _Pred& __pred) {
  constexpr size_t __unroll_count = 4;
  constexpr size_t __vec_size     = __native_vector_size<_Tp>;
  using __vec                     = __simd_vector<_Tp, __vec_size>;

  if (!__libcpp_is_constant_evaluated()) {
    auto __orig_first = __first;
    while (static_cast<size_t>(__last - __first) > __unroll_count * __vec_size) [[__unlikely__]] {
      __vec __cmp_res[__unroll_count];

      // Store the comparison results first to make sure the compiler is allowed to reorder any loads.
      for (size_t __i = 0; __i != __unroll_count; ++__i) {
        __cmp_res[__i] = std::__load_vector<__vec>(__first + __i * __vec_size) !=
                         std::__load_vector<__vec>(__first + __i * __vec_size + 1);
      }

      for (size_t __i = 0; __i != __unroll_count; ++__i) {
        if (!std::__all_of(__cmp_res[__i])) {
          auto __offset = __i * __vec_size + std::__find_first_not_set(__cmp_res[__i]);
          return __first + __offset;
        }
      }

      __first += __unroll_count * __vec_size;
    }

    // check the last 0-3 vectors
    while (static_cast<size_t>(__last - __first) > __vec_size) [[__unlikely__]] {
      if (auto __cmp_res = std::__load_vector<__vec>(__first) != std::__load_vector<__vec>(__first + 1);
          !std::__all_of(__cmp_res)) {
        auto __offset = std::__find_first_not_set(__cmp_res);
        return __first + __offset;
      }
      __first += __vec_size;
    }

    if (__first == __last)
      return __first;

    // Check if we can load elements in front of the current pointer. If that's the case load a vector at
    // (last - vector_size - 1) to check the remaining elements
    if (static_cast<size_t>(__last - __orig_first) > __vec_size) {
      __first = __last - __vec_size - 1;
      auto __offset =
          std::__find_first_not_set(std::__load_vector<__vec>(__first) != std::__load_vector<__vec>(__first + 1));
      if (__offset == __vec_size)
        return __last;
      return __first + __offset;
    }
  } // else loop over the elements individually
  return std::__adjacent_find_loop(__first, __last, __pred);
}

template <class _Tp,
          class _Pred,
          __enable_if_t<is_integral<_Tp>::value && __desugars_to_v<__equal_tag, _Pred, _Tp, _Tp>, int> = 0>
[[__nodiscard__]] _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX20 _Tp*
__adjacent_find(_Tp* __first, _Tp* __last, _Pred& __pred) {
  return std::__adjacent_find_vectorized(__first, __last, __pred);
}

#endif // _LIBCPP_VECTORIZE_ALGORITHMS

template <class _ForwardIterator, class _BinaryPredicate>
[[__nodiscard__]] inline _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX20 _ForwardIterator
adjacent_find(_ForwardIterator __first, _ForwardIterator __last, _BinaryPredicate __pred) {
  return std::__rewrap_iter(
      __first, std::__adjacent_find(std::__unwrap_iter(__first), std::__unwrap_iter(__last), __pred));
}

template <class _ForwardIterator>
[[__nodiscard__]] inline _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX20 _ForwardIterator
adjacent_find(_ForwardIterator __first, _ForwardIterator __last) {
  return std::adjacent_find(std::move(__first), std::move(__last), __equal_to());
}

_LIBCPP_END_NAMESPACE_STD

_LIBCPP_POP_MACROS

#endif // _LIBCPP___ALGORITHM_ADJACENT_FIND_H
