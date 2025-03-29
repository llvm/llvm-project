//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___ALGORITHM_FILL_FILL_N_COMMON_H
#define _LIBCPP___ALGORITHM_FILL_FILL_N_COMMON_H

#include <__algorithm/for_each_segment.h>
#include <__algorithm/min.h>
#include <__config>
#include <__fwd/bit_reference.h>
#include <__iterator/iterator_traits.h>
#include <__iterator/next.h>
#include <__iterator/segmented_iterator.h>
#include <__memory/pointer_traits.h>
#include <__type_traits/enable_if.h>
#include <__utility/convert_to_integral.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_PUSH_MACROS
#include <__undef_macros>

_LIBCPP_BEGIN_NAMESPACE_STD

template <class _ForwardIterator,
          class _Sentinel,
          class _Tp,
          __enable_if_t<__has_forward_iterator_category<_ForwardIterator>::value, int> = 0>
_LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX20 void
__fill(_ForwardIterator __first, _Sentinel __last, const _Tp& __value);

template <class _RandomAccessIterator,
          class _Tp,
          __enable_if_t<__has_random_access_iterator_category<_RandomAccessIterator>::value &&
                            !__is_segmented_iterator<_RandomAccessIterator>::value,
                        int> = 0>
_LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX20 void
__fill(_RandomAccessIterator __first, _RandomAccessIterator __last, const _Tp& __value);

template <class _SegmentedIterator,
          class _Tp,
          __enable_if_t<__is_segmented_iterator<_SegmentedIterator>::value, int> = 0>
_LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX20 void
__fill(_SegmentedIterator __first, _SegmentedIterator __last, const _Tp& __value);

template <class _OutIter, class _Size, class _Tp, __enable_if_t<!__is_segmented_iterator<_OutIter>::value, int> = 0>
inline _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX20 _OutIter
__fill_n(_OutIter __first, _Size __n, const _Tp& __value) {
  for (; __n > 0; ++__first, (void)--__n)
    *__first = __value;
  return __first;
}

template <bool _FillVal, class _Cp>
_LIBCPP_CONSTEXPR_SINCE_CXX20 _LIBCPP_HIDE_FROM_ABI void
__fill_n_bool(__bit_iterator<_Cp, false> __first, typename __size_difference_type_traits<_Cp>::size_type __n) {
  using _It            = __bit_iterator<_Cp, false>;
  using __storage_type = typename _It::__storage_type;

  const int __bits_per_word = _It::__bits_per_word;
  // do first partial word
  if (__first.__ctz_ != 0) {
    __storage_type __clz_f = static_cast<__storage_type>(__bits_per_word - __first.__ctz_);
    __storage_type __dn    = std::min(__clz_f, __n);
    std::__fill_masked_range(std::__to_address(__first.__seg_), __clz_f - __dn, __first.__ctz_, _FillVal);
    __n -= __dn;
    ++__first.__seg_;
  }
  // do middle whole words
  __storage_type __nw = __n / __bits_per_word;
  std::__fill_n(std::__to_address(__first.__seg_), __nw, _FillVal ? static_cast<__storage_type>(-1) : 0);
  __n -= __nw * __bits_per_word;
  // do last partial word
  if (__n > 0) {
    __first.__seg_ += __nw;
    std::__fill_masked_range(std::__to_address(__first.__seg_), __bits_per_word - __n, 0u, _FillVal);
  }
}

template <class _Cp, class _Size>
inline _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX20 __bit_iterator<_Cp, false>
__fill_n(__bit_iterator<_Cp, false> __first, _Size __n, const bool& __value) {
  if (__n > 0) {
    if (__value)
      std::__fill_n_bool<true>(__first, __n);
    else
      std::__fill_n_bool<false>(__first, __n);
  }
  return __first + __n;
}

template < class _OutIter,
           class _Size,
           class _Tp,
           __enable_if_t<__is_segmented_iterator<_OutIter>::value && __has_forward_iterator_category<_OutIter>::value,
                         int> = 0>
inline _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX14 _OutIter
__fill_n(_OutIter __first, _Size __n, const _Tp& __value) {
  _OutIter __last = std::next(__first, __n);
  std::__fill(__first, __last, __value);
  return __last;
}

template <class _ForwardIterator,
          class _Sentinel,
          class _Tp,
          __enable_if_t<__has_forward_iterator_category<_ForwardIterator>::value, int> >
inline _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX20 void
__fill(_ForwardIterator __first, _Sentinel __last, const _Tp& __value) {
  for (; __first != __last; ++__first)
    *__first = __value;
}

template <class _OutIter, class _Tp>
struct _FillSegment {
  using _Traits _LIBCPP_NODEBUG = __segmented_iterator_traits<_OutIter>;

  const _Tp& __value_;

  _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX14 explicit _FillSegment(const _Tp& __value) : __value_(__value) {}

  _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX14 void
  operator()(typename _Traits::__local_iterator __lfirst, typename _Traits::__local_iterator __llast) {
    std::__fill(__lfirst, __llast, __value_);
  }
};

template <class _RandomAccessIterator,
          class _Tp,
          __enable_if_t<__has_random_access_iterator_category<_RandomAccessIterator>::value &&
                            !__is_segmented_iterator<_RandomAccessIterator>::value,
                        int> >
inline _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX20 void
__fill(_RandomAccessIterator __first, _RandomAccessIterator __last, const _Tp& __value) {
  std::__fill_n(__first, __last - __first, __value);
}

template <class _SegmentedIterator, class _Tp, __enable_if_t<__is_segmented_iterator<_SegmentedIterator>::value, int> >
inline _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX20 void
__fill(_SegmentedIterator __first, _SegmentedIterator __last, const _Tp& __value) {
  std::__for_each_segment(__first, __last, _FillSegment<_SegmentedIterator, _Tp>(__value));
}

_LIBCPP_END_NAMESPACE_STD

_LIBCPP_POP_MACROS

#endif // _LIBCPP___ALGORITHM_FILL_FILL_N_COMMON_H
