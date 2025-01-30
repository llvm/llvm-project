//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___ALGORITHM_COPY_H
#define _LIBCPP___ALGORITHM_COPY_H

#include <__algorithm/copy_move_common.h>
#include <__algorithm/for_each_segment.h>
#include <__algorithm/min.h>
#include <__config>
#include <__fwd/bit_reference.h>
#include <__iterator/distance.h>
#include <__iterator/iterator_traits.h>
#include <__iterator/segmented_iterator.h>
#include <__type_traits/common_type.h>
#include <__type_traits/enable_if.h>
#include <__type_traits/is_convertible.h>
#include <__utility/move.h>
#include <__utility/pair.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_PUSH_MACROS
#include <__undef_macros>

_LIBCPP_BEGIN_NAMESPACE_STD

template <class _InIter, class _Sent, class _OutIter>
inline _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX14 pair<_InIter, _OutIter> __copy(_InIter, _Sent, _OutIter);

template <class _InIter,
          class _Sent,
          __enable_if_t<__has_input_iterator_category<_InIter>::value &&
                            !__has_random_access_iterator_category<_InIter>::value,
                        int> = 0>
_LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX17 typename iterator_traits<_InIter>::difference_type
__iter_sent_distance(_InIter __first, _Sent __last) {
  typename iterator_traits<_InIter>::difference_type __r(0);
  for (; __first != __last; ++__first)
    ++__r;
  return __r;
}

template <class _InIter, class _Sent, __enable_if_t<__has_random_access_iterator_category<_InIter>::value, int> = 0>
_LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX17 typename iterator_traits<_InIter>::difference_type
__iter_sent_distance(_InIter __first, _Sent __last) {
  return static_cast<typename iterator_traits<_InIter>::difference_type>(__last - __first);
}

struct __copy_impl {
  template <class _InIter, class _Sent, class _OutIter>
  _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX14 pair<_InIter, _OutIter>
  operator()(_InIter __first, _Sent __last, _OutIter __result) const {
    while (__first != __last) {
      *__result = *__first;
      ++__first;
      ++__result;
    }

    return std::make_pair(std::move(__first), std::move(__result));
  }

  template <class _InIter, class _OutIter>
  struct _CopySegment {
    using _Traits _LIBCPP_NODEBUG = __segmented_iterator_traits<_InIter>;

    _OutIter& __result_;

    _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX14 explicit _CopySegment(_OutIter& __result)
        : __result_(__result) {}

    _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX14 void
    operator()(typename _Traits::__local_iterator __lfirst, typename _Traits::__local_iterator __llast) {
      __result_ = std::__copy(__lfirst, __llast, std::move(__result_)).second;
    }
  };

  template <class _InIter, class _OutIter, __enable_if_t<__is_segmented_iterator<_InIter>::value, int> = 0>
  _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX14 pair<_InIter, _OutIter>
  operator()(_InIter __first, _InIter __last, _OutIter __result) const {
    std::__for_each_segment(__first, __last, _CopySegment<_InIter, _OutIter>(__result));
    return std::make_pair(__last, std::move(__result));
  }

  template <class _InIter,
            class _OutIter,
            __enable_if_t<__has_random_access_iterator_category<_InIter>::value &&
                              !__is_segmented_iterator<_InIter>::value && __is_segmented_iterator<_OutIter>::value,
                          int> = 0>
  _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX14 pair<_InIter, _OutIter>
  operator()(_InIter __first, _InIter __last, _OutIter __result) const {
    using _Traits = __segmented_iterator_traits<_OutIter>;
    using _DiffT  = typename common_type<__iter_diff_t<_InIter>, __iter_diff_t<_OutIter> >::type;

    if (__first == __last)
      return std::make_pair(std::move(__first), std::move(__result));

    auto __local_first      = _Traits::__local(__result);
    auto __segment_iterator = _Traits::__segment(__result);
    while (true) {
      auto __local_last = _Traits::__end(__segment_iterator);
      auto __size       = std::min<_DiffT>(__local_last - __local_first, __last - __first);
      auto __iters      = std::__copy(__first, __first + __size, __local_first);
      __first           = std::move(__iters.first);

      if (__first == __last)
        return std::make_pair(std::move(__first), _Traits::__compose(__segment_iterator, std::move(__iters.second)));

      __local_first = _Traits::__begin(++__segment_iterator);
    }
  }

  template <class _InIter,
            class _Sent,
            class _Cp,
            __enable_if_t<(__has_forward_iterator_category<_InIter>::value ||
                           __has_iterator_concept_convertible_to<_InIter, forward_iterator_tag>::value) &&
                              is_convertible<typename iterator_traits<_InIter>::value_type, bool>::value,
                          int> = 0>
  _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX14 pair<_InIter, __bit_iterator<_Cp, false> >
  operator()(_InIter __first, _Sent __last, __bit_iterator<_Cp, false> __result) const {
    using _It            = __bit_iterator<_Cp, false>;
    using __storage_type = typename _It::__storage_type;
#if _LIBCPP_STD_VER >= 20
    __storage_type __n = static_cast<__storage_type>(std::ranges::distance(__first, __last));
#else
    __storage_type __n = static_cast<__storage_type>(std::__iter_sent_distance(__first, __last));
#endif
    const unsigned __bits_per_word = _It::__bits_per_word;

    if (__first != __last) {
      // do first partial word, if present
      if (__result.__ctz_ != 0) {
        __storage_type __clz = static_cast<__storage_type>(__bits_per_word - __result.__ctz_);
        __storage_type __dn  = std::min(__clz, __n);
        __storage_type __w   = *__result.__seg_;
        __storage_type __m   = (~__storage_type(0) << __result.__ctz_) & (~__storage_type(0) >> (__clz - __dn));
        __w &= ~__m;
        for (__storage_type __i = 0; __i < __dn; ++__i, ++__first)
          __w |= static_cast<__storage_type>(*__first) << __result.__ctz_++;
        *__result.__seg_ = __w;
        if (__result.__ctz_ == __bits_per_word) {
          __result.__ctz_ = 0;
          ++__result.__seg_;
        }
        __n -= __dn;
      }
    }
    // do middle whole words, if present
    __storage_type __nw = __n / __bits_per_word;
    __n -= __nw * __bits_per_word;
    for (; __nw; --__nw) {
      __storage_type __w = 0;
      for (__storage_type __i = 0; __i < __bits_per_word; ++__i, ++__first)
        __w |= static_cast<__storage_type>(*__first) << __i;
      *__result.__seg_++ = __w;
    }
    // do last partial word, if present
    if (__n) {
      __storage_type __w = 0;
      for (__storage_type __i = 0; __i < __n; ++__i, ++__first)
        __w |= static_cast<__storage_type>(*__first) << __i;
      __storage_type __m = ~__storage_type(0) >> (__bits_per_word - __n);
      *__result.__seg_ &= ~__m;
      *__result.__seg_ |= __w;
      __result.__ctz_ = __n;
    }
    return std::make_pair(std::move(__first), std::move(__result));
  }

  // At this point, the iterators have been unwrapped so any `contiguous_iterator` has been unwrapped to a pointer.
  template <class _In, class _Out, __enable_if_t<__can_lower_copy_assignment_to_memmove<_In, _Out>::value, int> = 0>
  _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX14 pair<_In*, _Out*>
  operator()(_In* __first, _In* __last, _Out* __result) const {
    return std::__copy_trivial_impl(__first, __last, __result);
  }
};

template <class _InIter, class _Sent, class _OutIter>
pair<_InIter, _OutIter> inline _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX14
__copy(_InIter __first, _Sent __last, _OutIter __result) {
  return std::__copy_move_unwrap_iters<__copy_impl>(std::move(__first), std::move(__last), std::move(__result));
}

template <class _InputIterator, class _OutputIterator>
inline _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX20 _OutputIterator
copy(_InputIterator __first, _InputIterator __last, _OutputIterator __result) {
  return std::__copy(__first, __last, __result).second;
}

_LIBCPP_END_NAMESPACE_STD

_LIBCPP_POP_MACROS

#endif // _LIBCPP___ALGORITHM_COPY_H
