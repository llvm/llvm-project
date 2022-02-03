// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___ALGORITHM_IN_OUT_OUT_RESULT_H
#define _LIBCPP___ALGORITHM_IN_OUT_OUT_RESULT_H

#include <__concepts/convertible_to.h>
#include <__config>
#include <__utility/move.h>

_LIBCPP_BEGIN_NAMESPACE_STD

#ifndef _LIBCPP_HAS_NO_CONCEPTS

namespace ranges {
template <class _I1, class _O1, class _O2>
struct in_out_out_result {
  [[no_unique_address]] _I1 in;
  [[no_unique_address]] _O1 out1;
  [[no_unique_address]] _O2 out2;

  template <class _II1, class _OO1, class _OO2>
    requires convertible_to<const _I1&, _II1> && convertible_to<const _O1&, _OO1> && convertible_to<const _O2&, _OO2>
  _LIBCPP_HIDE_FROM_ABI constexpr
  operator in_out_out_result<_II1, _OO1, _OO2>() const& {
    return {in, out1, out2};
  }

  template <class _II1, class _OO1, class _OO2>
    requires convertible_to<_I1, _II1> && convertible_to<_O1, _OO1> && convertible_to<_O2, _OO2>
  _LIBCPP_HIDE_FROM_ABI constexpr
  operator in_out_out_result<_II1, _OO1, _OO2>() && {
    return {_VSTD::move(in), _VSTD::move(out1), _VSTD::move(out2)};
  }
};
} // namespace ranges

#endif // _LIBCPP_HAS_NO_CONCEPTS

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___ALGORITHM_IN_OUT_OUT_RESULT_H
