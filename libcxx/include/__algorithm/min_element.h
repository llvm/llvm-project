//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___ALGORITHM_MIN_ELEMENT_H
#define _LIBCPP___ALGORITHM_MIN_ELEMENT_H

#include <__algorithm/comp.h>
#include <__algorithm/comp_ref_type.h>
#include <__algorithm/iterator_operations.h>
#include <__config>
#include <__functional/identity.h>
#include <__iterator/iterator_traits.h>
#include <__type_traits/invoke.h>
#include <__type_traits/is_callable.h>
#include <__utility/move.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_PUSH_MACROS
#include <__undef_macros>

_LIBCPP_BEGIN_NAMESPACE_STD

template <class _Iter, class _Sent, class = void>
struct __ConstTimeDistance : false_type {};

#if _LIBCPP_STD_VER >= 20

template <class _Iter, class _Sent>
struct __ConstTimeDistance< _Iter, _Sent, __enable_if_t< sized_sentinel_for<_Sent, _Iter> >> : true_type {};

#else

template <class _Iter>
struct __ConstTimeDistance<
    _Iter,
    _Iter,
    __enable_if_t< is_same<typename iterator_traits<_Iter>::iterator_category, random_access_iterator_tag>::value> >
    : true_type {};

#endif // _LIBCPP_STD_VER >= 20

template <class _Comp, class _Iter, class _Sent, class _Proj>
inline _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX14 _Iter __min_element(
    _Iter __first,
    _Sent __last,
    _Comp __comp,
    _Proj& __proj,
    /*_ConstTimeDistance*/ false_type) {
  if (__first == __last)
    return __first;

  _Iter __i = __first;
  while (++__i != __last)
    if (std::__invoke(__comp, std::__invoke(__proj, *__i), std::__invoke(__proj, *__first)))
      __first = __i;
  return __first;
}

template <class _Comp, class _Iter, class _Sent, class _Proj>
inline _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX14 _Iter __min_element(
    _Iter __first,
    _Sent __last,
    _Comp __comp,
    _Proj& __proj,
    /*ConstTimeDistance*/ true_type) {
  if (__first == __last)
    return __first;

  typedef typename std::iterator_traits<_Iter>::difference_type diff_type;
  diff_type __n = std::distance(__first, __last);

  if (__n <= 64) {
    _Iter __i = __first;
    while (++__i != __last)
      if (std::__invoke(__comp, std::__invoke(__proj, *__i), std::__invoke(__proj, *__first)))
        __first = __i;
    return __first;
  }

  diff_type __block_size = 256;

  diff_type __n_blocked = __n - (__n % __block_size);
  _Iter __block_start = __first, __block_end = __first;

  typedef typename std::iterator_traits<_Iter>::value_type value_type;
  value_type __min_val = std::__invoke(__proj, *__first);

  _Iter __curr = __first;
  for (diff_type __i = 0; __i < __n_blocked; __i += __block_size) {
    _Iter __start          = __curr;
    value_type __block_min = __min_val;
    for (diff_type __j = 0; __j < __block_size; __j++) {
      if (std::__invoke(__comp, std::__invoke(__proj, *__curr), __block_min)) {
        __block_min = *__curr;
      }
      __curr++;
    }
    if (std::__invoke(__comp, __block_min, __min_val)) {
      __min_val     = __block_min;
      __block_start = __start;
      __block_end   = __curr;
    }
  }

  value_type __epilogue_min = __min_val;
  _Iter __epilogue_start    = __curr;
  while (__curr != __last) {
    if (std::__invoke(__comp, std::__invoke(__proj, *__curr), __epilogue_min)) {
      __epilogue_min = *__curr;
    }
    __curr++;
  }
  if (std::__invoke(__comp, __epilogue_min, __min_val)) {
    __min_val     = __epilogue_min;
    __block_start = __epilogue_start;
    __block_end   = __last;
  }

  for (; __block_start != __block_end; ++__block_start)
    if (std::__invoke(__proj, *__block_start) == __min_val)
      break;
  return __block_start;
}

template <class _Comp, class _Iter, class _Sent, class _Proj>
inline _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX14 _Iter
__min_element(_Iter __first, _Sent __last, _Comp __comp, _Proj& __proj) {
  return std::__min_element<_Comp>(
      std::move(__first), std::move(__last), __comp, __proj, __ConstTimeDistance<_Iter, _Sent>());
}

template <class _Comp, class _Iter, class _Sent>
_LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX14 _Iter __min_element(_Iter __first, _Sent __last, _Comp __comp) {
  auto __proj = __identity();
  return std::__min_element<_Comp>(std::move(__first), std::move(__last), __comp, __proj);
}

template <class _ForwardIterator, class _Compare>
[[__nodiscard__]] inline _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX14 _ForwardIterator
min_element(_ForwardIterator __first, _ForwardIterator __last, _Compare __comp) {
  static_assert(
      __has_forward_iterator_category<_ForwardIterator>::value, "std::min_element requires a ForwardIterator");
  static_assert(
      __is_callable<_Compare&, decltype(*__first), decltype(*__first)>::value, "The comparator has to be callable");

  return std::__min_element<__comp_ref_type<_Compare> >(std::move(__first), std::move(__last), __comp);
}

template <class _ForwardIterator>
[[__nodiscard__]] inline _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX14 _ForwardIterator
min_element(_ForwardIterator __first, _ForwardIterator __last) {
  return std::min_element(__first, __last, __less<>());
}

_LIBCPP_END_NAMESPACE_STD

_LIBCPP_POP_MACROS

#endif // _LIBCPP___ALGORITHM_MIN_ELEMENT_H
