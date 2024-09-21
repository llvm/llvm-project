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

#include <__config>
#include <__iterator/concepts.h>
#include <__ranges/access.h>
#include <__ranges/concepts.h>
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
template <input_or_output_iterator O, sentinel_for<O> S, weakly_incrementable T>
  requires indirectly_writable<O, const T&>
constexpr pair<O, T> iota(O first, S last, T value) {
  while (first != last) {
    *first = value;
    ++first;
    ++value;
  }
  return {std::move(first), std::move(value)};
}

template < input_or_output_iterator O, weakly_incrementable T, output_range<const T&> R >
constexpr pair<O, T> iota(R&& r, T value) {
  return std::ranges::iota(std::ranges::begin(r), std::ranges::end(r), std::move(value));
}
} // namespace ranges
#endif // _LIBCPP_STD_VER >= 23

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___NUMERIC_IOTA_H
