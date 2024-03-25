// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___ALGORITHM_MISMATCH_H
#define _LIBCPP___ALGORITHM_MISMATCH_H

#include <__algorithm/comp.h>
#include <__algorithm/simd_utils.h>
#include <__algorithm/unwrap_iter.h>
#include <__config>
#include <__functional/identity.h>
#include <__type_traits/invoke.h>
#include <__type_traits/is_constant_evaluated.h>
#include <__type_traits/is_equality_comparable.h>
#include <__type_traits/operation_traits.h>
#include <__utility/move.h>
#include <__utility/pair.h>
#include <__utility/unreachable.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_PUSH_MACROS
#include <__undef_macros>

_LIBCPP_BEGIN_NAMESPACE_STD

template <class _Iter1, class _Sent1, class _Iter2, class _Pred, class _Proj1, class _Proj2>
_LIBCPP_NODISCARD _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX20 pair<_Iter1, _Iter2>
__mismatch_loop(_Iter1 __first1, _Sent1 __last1, _Iter2 __first2, _Pred& __pred, _Proj1& __proj1, _Proj2& __proj2) {
  while (__first1 != __last1) {
    if (!std::__invoke(__pred, std::__invoke(__proj1, *__first1), std::__invoke(__proj2, *__first2)))
      break;
    ++__first1;
    ++__first2;
  }
  return std::make_pair(std::move(__first1), std::move(__first2));
}

template <class _Iter1, class _Sent1, class _Iter2, class _Pred, class _Proj1, class _Proj2>
_LIBCPP_NODISCARD _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX20 pair<_Iter1, _Iter2>
__mismatch(_Iter1 __first1, _Sent1 __last1, _Iter2 __first2, _Pred& __pred, _Proj1& __proj1, _Proj2& __proj2) {
  return std::__mismatch_loop(__first1, __last1, __first2, __pred, __proj1, __proj2);
}

#if _LIBCPP_VECTORIZE_ALGORITHMS

template <class _Tp,
          class _Pred,
          class _Proj1,
          class _Proj2,
          __enable_if_t<is_integral<_Tp>::value && __desugars_to<__equal_tag, _Pred, _Tp, _Tp>::value &&
                            __is_identity<_Proj1>::value && __is_identity<_Proj2>::value,
                        int> = 0>
_LIBCPP_NODISCARD _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX20 pair<_Tp*, _Tp*>
__mismatch(_Tp* __first1, _Tp* __last1, _Tp* __first2, _Pred& __pred, _Proj1& __proj1, _Proj2& __proj2) {
  constexpr size_t __unroll_count = 4;
  constexpr size_t __vec_size     = __native_vector_size<_Tp>;
  using __vec                     = __simd_vector<_Tp, __vec_size>;
  if (!__libcpp_is_constant_evaluated()) {
    while (static_cast<size_t>(__last1 - __first1) >= __unroll_count * __vec_size) [[__unlikely__]] {
      __vec __lhs[__unroll_count];
      __vec __rhs[__unroll_count];

      for (size_t __i = 0; __i != __unroll_count; ++__i) {
        __lhs[__i] = std::__load_vector<__vec>(__first1 + __i * __vec_size);
        __rhs[__i] = std::__load_vector<__vec>(__first2 + __i * __vec_size);
      }

      for (size_t __i = 0; __i != __unroll_count; ++__i) {
        if (auto __cmp_res = __lhs[__i] == __rhs[__i]; !std::__all_of(__cmp_res)) {
          auto __offset = __i * __vec_size + std::__find_first_not_set(__cmp_res);
          return {__first1 + __offset, __first2 + __offset};
        }
      }

      __first1 += __unroll_count * __vec_size;
      __first2 += __unroll_count * __vec_size;
    }
  }
  // TODO: Consider vectorizing the tail
  return std::__mismatch_loop(__first1, __last1, __first2, __pred, __proj1, __proj2);
}

#endif // _LIBCPP_VECTORIZE_ALGORITHMS

template <class _InputIterator1, class _InputIterator2, class _BinaryPredicate>
_LIBCPP_NODISCARD_EXT inline _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX20 pair<_InputIterator1, _InputIterator2>
mismatch(_InputIterator1 __first1, _InputIterator1 __last1, _InputIterator2 __first2, _BinaryPredicate __pred) {
  __identity __proj;
  auto __res = std::__mismatch(
      std::__unwrap_iter(__first1), std::__unwrap_iter(__last1), std::__unwrap_iter(__first2), __pred, __proj, __proj);
  return std::make_pair(std::__rewrap_iter(__first1, __res.first), std::__rewrap_iter(__first2, __res.second));
}

template <class _InputIterator1, class _InputIterator2>
_LIBCPP_NODISCARD_EXT inline _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX20 pair<_InputIterator1, _InputIterator2>
mismatch(_InputIterator1 __first1, _InputIterator1 __last1, _InputIterator2 __first2) {
  return std::mismatch(__first1, __last1, __first2, __equal_to());
}

#if _LIBCPP_STD_VER >= 14
template <class _InputIterator1, class _InputIterator2, class _BinaryPredicate>
_LIBCPP_NODISCARD_EXT inline _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX20 pair<_InputIterator1, _InputIterator2>
mismatch(_InputIterator1 __first1,
         _InputIterator1 __last1,
         _InputIterator2 __first2,
         _InputIterator2 __last2,
         _BinaryPredicate __pred) {
  for (; __first1 != __last1 && __first2 != __last2; ++__first1, (void)++__first2)
    if (!__pred(*__first1, *__first2))
      break;
  return pair<_InputIterator1, _InputIterator2>(__first1, __first2);
}

template <class _InputIterator1, class _InputIterator2>
_LIBCPP_NODISCARD_EXT inline _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX20 pair<_InputIterator1, _InputIterator2>
mismatch(_InputIterator1 __first1, _InputIterator1 __last1, _InputIterator2 __first2, _InputIterator2 __last2) {
  return std::mismatch(__first1, __last1, __first2, __last2, __equal_to());
}
#endif

_LIBCPP_END_NAMESPACE_STD

_LIBCPP_POP_MACROS

#endif // _LIBCPP___ALGORITHM_MISMATCH_H
