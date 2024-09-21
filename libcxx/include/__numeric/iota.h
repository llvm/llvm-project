// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___NUMERIC_IOTA_H
#define _LIBCPP___NUMERIC_IOTA_H

#include <__algorithm/out_value_result.h>
#include <__config>
#include <__iterator/concepts.h>
#include <__ranges/access.h>
#include <__ranges/concepts.h>
#include <__utility/as_const.h>
#include <__utility/pair.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

template <class _ForwardIterator, class _Tp>
_LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX20 void
iota(_ForwardIterator __first, _ForwardIterator __last, _Tp __value) {
  for (; __first != __last; ++__first, (void)++__value)
    *__first = __value;
}

#if _LIBCPP_STD_VER >= 23
namespace ranges {
template < class _O, class _T >
using iota_result = out_value_result<_O, _T>;

template <input_or_output_iterator _O, sentinel_for<_O> _S, weakly_incrementable _T>
  requires indirectly_writable<_O, const _T&>
constexpr iota_result<_O, _T> iota(_O __first, _S __last, _T __value) {
  while (__first != __last) {
    *__first = std::as_const(__value);
    ++__first;
    ++__value;
  }
  return {std::move(__first), std::move(__value)};
}

template <weakly_incrementable _T, output_range<const _T&> _R >
constexpr iota_result<borrowed_iterator_t<_R>, _T> iota(_R&& __r, _T __value) {
  return std::ranges::iota(std::ranges::begin(__r), std::ranges::end(__r), std::move(__value));
}
} // namespace ranges
#endif // _LIBCPP_STD_VER >= 23

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___NUMERIC_IOTA_H
