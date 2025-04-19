//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___ALGORITHM_MINMAX_ELEMENT_H
#define _LIBCPP___ALGORITHM_MINMAX_ELEMENT_H

#include <__algorithm/comp.h>
#include <__algorithm/simd_utils.h>
#include <__algorithm/unwrap_iter.h>
#include <__config>
#include <__functional/identity.h>
#include <__iterator/iterator_traits.h>
#include <__type_traits/invoke.h>
#include <__type_traits/is_callable.h>
#include <__type_traits/is_integral.h>
#include <__utility/pair.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

template <class _Comp, class _Proj>
class _MinmaxElementLessFunc {
  _Comp& __comp_;
  _Proj& __proj_;

public:
  _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR _MinmaxElementLessFunc(_Comp& __comp, _Proj& __proj)
      : __comp_(__comp), __proj_(__proj) {}

  template <class _Iter>
  _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX14 bool operator()(_Iter& __it1, _Iter& __it2) {
    return std::__invoke(__comp_, std::__invoke(__proj_, *__it1), std::__invoke(__proj_, *__it2));
  }
};

template<class _Iter, class _Sent, class _Proj, class _Comp>
_LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX14 pair<_Iter, _Iter> 
__minmax_element_loop(_Iter __first, _Sent __last, _Comp& __comp, _Proj& __proj) {
  auto __less = _MinmaxElementLessFunc<_Comp, _Proj>(__comp, __proj);

  pair<_Iter, _Iter> __result(__first, __first);
  if (__first == __last || ++__first == __last)
    return __result;

  if (__less(__first, __result.first))
    __result.first = __first;
  else
    __result.second = __first;

  while (++__first != __last) {
    _Iter __i = __first;
    if (++__first == __last) {
      if (__less(__i, __result.first))
        __result.first = __i;
      else if (!__less(__i, __result.second))
        __result.second = __i;
      return __result;
    }

    if (__less(__first, __i)) {
      if (__less(__first, __result.first))
        __result.first = __first;
      if (!__less(__i, __result.second))
        __result.second = __i;
    } else {
      if (__less(__i, __result.first))
        __result.first = __i;
      if (!__less(__first, __result.second))
        __result.second = __first;
    }
  }

  return __result;
}

#if _LIBCPP_VECTORIZE_ALGORITHMS
template<class _Iter>
_LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX14 pair<_Iter, _Iter> 
__minmax_element_vectorized(_Iter __first, _Iter __last) {
  using __value_type              = __iter_value_type<_Iter>;
  constexpr size_t __unroll_count = 4;
  constexpr size_t __vec_size     = __native_vector_size<__value_type>;
  using __vec_type                = __simd_vector<__value_type, __vec_size>;

  auto __comp = std::__less<>{};
  std::__identity __proj;
  if (static_cast<size_t>(__last - __first) < __vec_size) [[__unlikely__]] {
    return std::__minmax_element_loop(__first, __last, __comp, __proj);
  }

  __value_type __min_element = *__first;
  __value_type __max_element = *__first;

  _Iter __min_block_start = __first;
  _Iter __min_block_end = __last + 1;
  _Iter __max_block_start = __first;
  _Iter __max_block_end = __last + 1;
  
  while(static_cast<size_t>(__last - __first) >= __unroll_count * __vec_size) [[__likely__]]{
    __vec_type __vec[__unroll_count];
    for(size_t __i = 0; __i < __unroll_count; ++__i) {
      __vec[__i] = std::__load_vector<__vec_type>(__first + __i * __vec_size);
      // block min
      auto __block_min_element = __builtin_reduce_min(__vec[__i]);
      if (__block_min_element < __min_element) {
        __min_element = __block_min_element;
        __min_block_start = __first + __i * __vec_size;
        __min_block_end = __first + (__i + 1) * __vec_size;
      }
      // block max
      auto __block_max_element = __builtin_reduce_max(__vec[__i]);
      if (__block_max_element >= __max_element) {
        __max_element = __block_max_element;
        __max_block_start = __first + __i * __vec_size;
        __max_block_end = __first + (__i + 1) * __vec_size;
      }
    }
    __first += __unroll_count * __vec_size;
  }

  // remaining vectors 
  while(static_cast<size_t>(__last - __first) >=  __vec_size) {
      __vec_type __vec = std::__load_vector<__vec_type>(__first);
      auto __block_min_element = __builtin_reduce_min(__vec);
      if (__block_min_element < __min_element) {
        __min_element = __block_min_element;
        __min_block_start = __first;
        __min_block_end = __first + __vec_size;
      }
      // max
      auto __block_max_element = __builtin_reduce_max(__vec);
      if (__block_max_element >= __max_element) {
        __max_element = __block_max_element;
        __max_block_start = __first;
        __max_block_end = __first + __vec_size;
      }
      __first += __vec_size;
  }

  if (__last > __first) {
    auto __epilogue = std::__minmax_element_loop(__first, __last, __comp, __proj);
    __value_type __epilogue_min_element = *__epilogue.first;
    __value_type __epilogue_max_element = *__epilogue.second;
    if (__epilogue_min_element < __min_element && __epilogue_max_element >= __max_element) {
      return __epilogue;
    } else if (__epilogue_min_element < __min_element) {
      __min_element = __epilogue_min_element;
      __min_block_start = __epilogue.first;
      __min_block_end   = __epilogue.first; // this is global min_element
    } else if (__epilogue_max_element >= __max_element) {
      __max_element = __epilogue_max_element;
      __max_block_start = __epilogue.second;
      __max_block_end   = __epilogue.second; // this is global max_element
    }
  }

  // locate min
  for (; __min_block_start != __min_block_end; ++__min_block_start) {
    __value_type __cur_min_element = *__min_block_start;
    if (__cur_min_element == __min_element)
      break;
  }

  // locate max
  for (_Iter __it = __max_block_start; __it != __max_block_end; ++__it) {
    __value_type __cur_max_element = *__it;
    if (__cur_max_element == __max_element)
      __max_block_start = __it;
  }

  return {__min_block_start, __max_block_start};
}

template <class _Iter, class _Proj, class _Comp,
          __enable_if_t
          <is_integral_v<__iter_value_type<_Iter>>
          && is_same_v<__iterator_category_type<_Iter>, random_access_iterator_tag>
          && __is_identity<_Proj>::value
          && __desugars_to_v<__less_tag, _Comp, _Iter, _Iter>,
          int> = 0
          >
_LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX14 pair<_Iter, _Iter>
__minmax_element_impl(_Iter __first, _Iter __last, _Comp& __comp, _Proj& __proj) {
  if (__libcpp_is_constant_evaluated()) {
    return __minmax_element_loop(__first, __last, __comp, __proj);
  } else {
    auto __res = std::__minmax_element_vectorized(std::__unwrap_iter(__first), std::__unwrap_iter(__last));
    return {std::__rewrap_iter(__first, __res.first), std::__rewrap_iter(__first, __res.second)};
  }
}
// template <class _Iter, class _Proj, class _Comp,
//           __enable_if_t
//           <!is_integral_v<__iter_value_type<_Iter>>
//           && is_same_v<__iterator_category_type<_Iter>, random_access_iterator_tag>
//           && __can_map_to_integer_v<__iter_value_type<_Iter>> 
//           && __libcpp_is_trivially_equality_comparable<__iter_value_type<_Iter>, __iter_value_type<_Iter>>::value
//           && __is_identity<_Proj>::value 
//           && __desugars_to_v<__less_tag, _Comp, _Iter, _Iter>,
//           int> = 0
//         >
// _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX14 pair<_Iter, _Iter>
// __minmax_element_impl(_Iter __first, _Iter __last, _Comp& __comp, _Proj& __proj) {
//   if (__libcpp_is_constant_evaluated()) {
//     return __minmax_element_loop(__first, __last, __comp, __proj);
//   } else {
//   }
// }
#endif // _LIBCPP_VECTORIZE_ALGORITHMS

template <class _Iter, class _Sent, class _Proj, class _Comp>
_LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX14 pair<_Iter, _Iter>
__minmax_element_impl(_Iter __first, _Sent __last, _Comp& __comp, _Proj& __proj) {
  return std::__minmax_element_loop(__first, __last, __comp, __proj);
}

template <class _ForwardIterator, class _Compare>
[[__nodiscard__]] _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX14 pair<_ForwardIterator, _ForwardIterator>
minmax_element(_ForwardIterator __first, _ForwardIterator __last, _Compare __comp) {
  static_assert(
      __has_forward_iterator_category<_ForwardIterator>::value, "std::minmax_element requires a ForwardIterator");
  static_assert(
      __is_callable<_Compare&, decltype(*__first), decltype(*__first)>::value, "The comparator has to be callable");
  auto __proj = __identity();
  return std::__minmax_element_impl(__first, __last, __comp, __proj);
}

template <class _ForwardIterator>
[[__nodiscard__]] inline _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX14 pair<_ForwardIterator, _ForwardIterator>
minmax_element(_ForwardIterator __first, _ForwardIterator __last) {
  return std::minmax_element(__first, __last, __less<>());
}

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___ALGORITHM_MINMAX_ELEMENT_H
