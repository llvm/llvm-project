//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___ALGORITHM_LOWER_BOUND_H
#define _LIBCPP___ALGORITHM_LOWER_BOUND_H

#include <__algorithm/comp.h>
#include <__algorithm/half_positive.h>
#include <__algorithm/iterator_operations.h>
#include <__config>
#include <__functional/identity.h>
#include <__functional/invoke.h>
#include <__iterator/advance.h>
#include <__iterator/distance.h>
#include <__iterator/iterator_traits.h>
#include <__type_traits/is_callable.h>
#include <__type_traits/remove_reference.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

template <class _AlgPolicy, class _Iter, class _Type, class _Proj, class _Comp>
_LIBCPP_NODISCARD_EXT _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX20 _Iter __lower_bound_bisecting(
    _Iter __first,
    const _Type& __value,
    typename iterator_traits<_Iter>::difference_type __len,
    _Comp& __comp,
    _Proj& __proj) {
  while (__len != 0) {
    auto __l2 = std::__half_positive(__len);
    _Iter __m = __first;
    _IterOps<_AlgPolicy>::advance(__m, __l2);
    if (std::__invoke(__comp, std::__invoke(__proj, *__m), __value)) {
      __first = ++__m;
      __len -= __l2 + 1;
    } else {
      __len = __l2;
    }
  }
  return __first;
}

// One-sided binary search, aka meta binary search, has been in the public domain for decades, and has the general
// advantage of being Ω(1) rather than the classic algorithm's Ω(log(n)), with the downside of executing at most
// 2*(log(n)-1) comparisons vs the classic algorithm's exact log(n). There are two scenarios in which it really shines:
// the first one is when operating over non-random iterators, because the classic algorithm requires knowing the
// container's size upfront, which adds Ω(n) iterator increments to the complexity. The second one is when you're
// traversing the container in order, trying to fast-forward to the next value: in that case, the classic algorithm
// would yield Ω(n*log(n)) comparisons and, for non-random iterators, Ω(n^2) iterator increments, whereas the one-sided
// version will yield O(n) operations on both counts, with a Ω(log(n)) bound on the number of comparisons.
template <class _AlgPolicy, class _Iter, class _Sent, class _Type, class _Proj, class _Comp>
_LIBCPP_NODISCARD_EXT _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX20 _Iter
__lower_bound_onesided(_Iter __first, _Sent __last, const _Type& __value, _Comp& __comp, _Proj& __proj) {
  // static_assert(std::is_base_of<std::forward_iterator_tag, typename _IterOps<_AlgPolicy>::template
  // __iterator_category<_Iter>>::value,
  //       "lower_bound() is a multipass algorithm and requires forward iterator or better");

  using _Distance = typename iterator_traits<_Iter>::difference_type;
  for (_Distance __step = 1; __first != __last; __step <<= 1) {
    auto __it   = __first;
    auto __dist = __step - _IterOps<_AlgPolicy>::advance(__it, __step, __last);
    // once we reach the last range where needle can be we must start
    // looking inwards, bisecting that range
    if (__it == __last || !std::__invoke(__comp, std::__invoke(__proj, *__it), __value)) {
      return std::__lower_bound_bisecting<_AlgPolicy>(__first, __value, __dist, __comp, __proj);
    }
    // range not found, move forward!
    __first = std::move(__it);
  }
  return __first;
}

template <class _AlgPolicy, class _InputIter, class _Sent, class _Type, class _Proj, class _Comp>
_LIBCPP_NODISCARD_EXT inline _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX20 _InputIter __lower_bound(
    _InputIter __first, _Sent __last, const _Type& __value, _Comp& __comp, _Proj& __proj, std::input_iterator_tag) {
  return std::__lower_bound_onesided<_AlgPolicy>(__first, __last, __value, __comp, __proj);
}

template <class _AlgPolicy, class _RandIter, class _Sent, class _Type, class _Proj, class _Comp>
_LIBCPP_NODISCARD_EXT inline _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX20 _RandIter __lower_bound(
    _RandIter __first,
    _Sent __last,
    const _Type& __value,
    _Comp& __comp,
    _Proj& __proj,
    std::random_access_iterator_tag) {
  const auto __dist = _IterOps<_AlgPolicy>::distance(__first, __last);
  return std::__lower_bound_bisecting<_AlgPolicy>(__first, __value, __dist, __comp, __proj);
}

template <class _AlgPolicy, class _Iter, class _Sent, class _Type, class _Proj, class _Comp>
_LIBCPP_NODISCARD_EXT inline _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX20 _Iter
__lower_bound(_Iter __first, _Sent __last, const _Type& __value, _Comp&& __comp, _Proj&& __proj) {
  return std::__lower_bound<_AlgPolicy>(
      __first, __last, __value, __comp, __proj, typename _IterOps<_AlgPolicy>::template __iterator_category<_Iter>());
}

template <class _ForwardIterator, class _Tp, class _Compare>
_LIBCPP_NODISCARD_EXT inline _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX20
_ForwardIterator lower_bound(_ForwardIterator __first, _ForwardIterator __last, const _Tp& __value, _Compare __comp) {
  static_assert(__is_callable<_Compare, decltype(*__first), const _Tp&>::value,
                "The comparator has to be callable");
  auto __proj = std::__identity();
  return std::__lower_bound<_ClassicAlgPolicy>(__first, __last, __value, std::move(__comp), std::move(__proj));
}

template <class _ForwardIterator, class _Tp>
_LIBCPP_NODISCARD_EXT inline _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX20
_ForwardIterator lower_bound(_ForwardIterator __first, _ForwardIterator __last, const _Tp& __value) {
  return std::lower_bound(__first, __last, __value, __less<>());
}

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___ALGORITHM_LOWER_BOUND_H
