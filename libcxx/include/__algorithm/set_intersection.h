//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___ALGORITHM_SET_INTERSECTION_H
#define _LIBCPP___ALGORITHM_SET_INTERSECTION_H

#include <__algorithm/comp.h>
#include <__algorithm/comp_ref_type.h>
#include <__algorithm/iterator_operations.h>
#include <__algorithm/lower_bound.h>
#include <__config>
#include <__functional/identity.h>
#include <__iterator/iterator_traits.h>
#include <__iterator/next.h>
#include <__type_traits/is_same.h>
#include <__utility/exchange.h>
#include <__utility/move.h>
#include <__utility/swap.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_PUSH_MACROS
#include <__undef_macros>

_LIBCPP_BEGIN_NAMESPACE_STD

template <class _InIter1, class _InIter2, class _OutIter>
struct __set_intersection_result {
  _InIter1 __in1_;
  _InIter2 __in2_;
  _OutIter __out_;

  // need a constructor as C++03 aggregate init is hard
  _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX20
  __set_intersection_result(_InIter1&& __in_iter1, _InIter2&& __in_iter2, _OutIter&& __out_iter)
      : __in1_(std::move(__in_iter1)), __in2_(std::move(__in_iter2)), __out_(std::move(__out_iter)) {}
};

template <class _AlgPolicy, class _Compare, class _InIter1, class _Sent1, class _InIter2, class _Sent2, class _OutIter>
struct _LIBCPP_NODISCARD_EXT __set_intersector {
  _InIter1& __first1_;
  const _Sent1& __last1_;
  _InIter2& __first2_;
  const _Sent2& __last2_;
  _OutIter& __result_;
  _Compare& __comp_;
  bool __prev_advanced_ = true;

  _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX20 __set_intersector(
      _InIter1& __first1, _Sent1& __last1, _InIter2& __first2, _Sent2& __last2, _OutIter& __result, _Compare& __comp)
      : __first1_(__first1),
        __last1_(__last1),
        __first2_(__first2),
        __last2_(__last2),
        __result_(__result),
        __comp_(__comp) {}

  _LIBCPP_NODISCARD_EXT _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX20
      __set_intersection_result<_InIter1, _InIter2, _OutIter>
      operator()() {
    while (__first2_ != __last2_) {
      __advance1_and_maybe_add_result();
      if (__first1_ == __last1_)
        break;
      __advance2_and_maybe_add_result();
    }
    return __set_intersection_result<_InIter1, _InIter2, _OutIter>(
        _IterOps<_AlgPolicy>::next(std::move(__first1_), std::move(__last1_)),
        _IterOps<_AlgPolicy>::next(std::move(__first2_), std::move(__last2_)),
        std::move(__result_));
  }

private:
  // advance __iter to the first element in the range where !__comp_(__iter, __value)
  // add result if this is the second consecutive call without advancing
  // this method only works if you alternate calls between __advance1_and_maybe_add_result() and
  // __advance2_and_maybe_add_result()
  template <class _Iter, class _Sent, class _Value>
  _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX20 void
  __advance_and_maybe_add_result(_Iter& __iter, const _Sent& __sentinel, const _Value& __value) {
    static _LIBCPP_CONSTEXPR std::__identity __proj;
    // use one-sided binary search for improved algorithmic complexity bounds
    // understanding how we can use binary search and still respect complexity
    // guarantees is _not_ straightforward, so let me explain: the guarantee
    // is "at most 2*(N+M)-1 comparisons", and one-sided binary search will
    // necessarily overshoot depending on the position of the needle in the
    // haystack -- for instance, if we're searching for 3 in (1, 2, 3, 4),
    // we'll check if 3<1, then 3<2, then 3<4, and, finally, 3<3, for a total of
    // 4 comparisons, when linear search would have yielded 3. However,
    // because we won't need to perform the intervening reciprocal comparisons
    // (ie 1<3, 2<3, 4<3), that extra comparison doesn't run afoul of the
    // guarantee. Additionally, this type of scenario can only happen for match
    // distances of up to 5 elements, because 2*log2(8) is 6, and we'll still
    // be worse-off at position 5 of an 8-element set. From then onwards
    // these scenarios can't happen.
    // TL;DR: we'll be 1 comparison worse-off compared to the classic linear-
    // searching algorithm if matching position 3 of a set with 4 elements,
    // or position 5 if the set has 7 or 8 elements, but we'll never exceed
    // the complexity guarantees from the standard.
    _Iter __tmp = std::__lower_bound_onesided<_AlgPolicy>(__iter, __sentinel, __value, __comp_, __proj);
    std::swap(__tmp, __iter);
    __add_output_unless(__tmp != __iter);
  }

  // advance __first1_ to the first element in the range where !__comp_(*__first1_, *__first2_)
  // add result if neither __first1_ nor __first2_ advanced in the last attempt (meaning they are equal)
  _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX20 void __advance1_and_maybe_add_result() {
    __advance_and_maybe_add_result(__first1_, __last1_, *__first2_);
  }

  // advance __first2_ to the first element in the range where !__comp_(*__first2_, *__first1_)
  // add result if neither __first1_ nor __first2_ advanced in the last attempt (meaning they are equal)
  _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX20 void __advance2_and_maybe_add_result() {
    __advance_and_maybe_add_result(__first2_, __last2_, *__first1_);
  }

  _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX20 void __add_output_unless(bool __advanced) {
    if (__advanced | __prev_advanced_) {
      __prev_advanced_ = __advanced;
    } else {
      *__result_ = *__first1_;
      ++__result_;
      ++__first1_;
      ++__first2_;
      __prev_advanced_ = true;
    }
  }
};

// with forward iterators we can use binary search to skip over entries
template <class _AlgPolicy,
          class _Compare,
          class _InForwardIter1,
          class _Sent1,
          class _InForwardIter2,
          class _Sent2,
          class _OutIter>
_LIBCPP_NODISCARD_EXT _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX20
    __set_intersection_result<_InForwardIter1, _InForwardIter2, _OutIter>
    __set_intersection(
        _InForwardIter1 __first1,
        _Sent1 __last1,
        _InForwardIter2 __first2,
        _Sent2 __last2,
        _OutIter __result,
        _Compare&& __comp,
        std::forward_iterator_tag,
        std::forward_iterator_tag) {
  std::__set_intersector<_AlgPolicy, _Compare, _InForwardIter1, _Sent1, _InForwardIter2, _Sent2, _OutIter>
      __intersector(__first1, __last1, __first2, __last2, __result, __comp);
  return __intersector();
}

// input iterators are not suitable for multipass algorithms, so we stick to the classic single-pass version
template <class _AlgPolicy,
          class _Compare,
          class _InInputIter1,
          class _Sent1,
          class _InInputIter2,
          class _Sent2,
          class _OutIter>
_LIBCPP_NODISCARD_EXT _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX20
    __set_intersection_result<_InInputIter1, _InInputIter2, _OutIter>
    __set_intersection(
        _InInputIter1 __first1,
        _Sent1 __last1,
        _InInputIter2 __first2,
        _Sent2 __last2,
        _OutIter __result,
        _Compare&& __comp,
        std::input_iterator_tag,
        std::input_iterator_tag) {
  while (__first1 != __last1 && __first2 != __last2) {
    if (__comp(*__first1, *__first2))
      ++__first1;
    else {
      if (!__comp(*__first2, *__first1)) {
        *__result = *__first1;
        ++__result;
        ++__first1;
      }
      ++__first2;
    }
  }

  return std::__set_intersection_result<_InInputIter1, _InInputIter2, _OutIter>(
      _IterOps<_AlgPolicy>::next(std::move(__first1), std::move(__last1)),
      _IterOps<_AlgPolicy>::next(std::move(__first2), std::move(__last2)),
      std::move(__result));
}

template <class _AlgPolicy, class _Iter>
class __set_intersection_iter_category {
  template <class _It>
  using __cat = typename std::_IterOps<_AlgPolicy>::template __iterator_category<_It>;
  template <class _It>
  static __cat<_It> test(__cat<_It>*);
  template <class>
  static std::input_iterator_tag test(...);

public:
  using __type = decltype(test<_Iter>(nullptr));
};

template <class _AlgPolicy, class _Compare, class _InIter1, class _Sent1, class _InIter2, class _Sent2, class _OutIter>
_LIBCPP_NODISCARD_EXT _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX20
    __set_intersection_result<_InIter1, _InIter2, _OutIter>
    __set_intersection(
        _InIter1 __first1, _Sent1 __last1, _InIter2 __first2, _Sent2 __last2, _OutIter __result, _Compare&& __comp) {
  return std::__set_intersection<_AlgPolicy>(
      std::move(__first1),
      std::move(__last1),
      std::move(__first2),
      std::move(__last2),
      std::move(__result),
      __comp,
      typename std::__set_intersection_iter_category<_AlgPolicy, _InIter1>::__type(),
      typename std::__set_intersection_iter_category<_AlgPolicy, _InIter2>::__type());
}

template <class _InputIterator1, class _InputIterator2, class _OutputIterator, class _Compare>
inline _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX20 _OutputIterator set_intersection(
    _InputIterator1 __first1,
    _InputIterator1 __last1,
    _InputIterator2 __first2,
    _InputIterator2 __last2,
    _OutputIterator __result,
    _Compare __comp) {
  return std::__set_intersection<_ClassicAlgPolicy, __comp_ref_type<_Compare> >(
             std::move(__first1),
             std::move(__last1),
             std::move(__first2),
             std::move(__last2),
             std::move(__result),
             __comp)
      .__out_;
}

template <class _InputIterator1, class _InputIterator2, class _OutputIterator>
inline _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX20 _OutputIterator set_intersection(
    _InputIterator1 __first1,
    _InputIterator1 __last1,
    _InputIterator2 __first2,
    _InputIterator2 __last2,
    _OutputIterator __result) {
  return std::__set_intersection<_ClassicAlgPolicy>(
             std::move(__first1),
             std::move(__last1),
             std::move(__first2),
             std::move(__last2),
             std::move(__result),
             __less<>())
      .__out_;
}

_LIBCPP_END_NAMESPACE_STD

_LIBCPP_POP_MACROS

#endif // _LIBCPP___ALGORITHM_SET_INTERSECTION_H
