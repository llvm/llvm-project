// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___ALGORITHM_IN_OUT_RESULT_H
#define _LIBCPP___ALGORITHM_IN_OUT_RESULT_H

#include <__concepts/convertible_to.h>
#include <__config>
#include <__utility/move.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

#if _LIBCPP_STD_VER > 17 && !defined(_LIBCPP_HAS_NO_INCOMPLETE_RANGES)

namespace ranges {

template<class _I1, class _O1>
struct in_out_result {
  _LIBCPP_NO_UNIQUE_ADDRESS _I1 in;
  _LIBCPP_NO_UNIQUE_ADDRESS _O1 out;

  template <class _I2, class _O2>
    requires convertible_to<const _I1&, _I2> && convertible_to<const _O1&, _O2>
  _LIBCPP_HIDE_FROM_ABI
  constexpr operator in_out_result<_I2, _O2>() const & {
    return {in, out};
  }

  template <class _I2, class _O2>
    requires convertible_to<_I1, _I2> && convertible_to<_O1, _O2>
  _LIBCPP_HIDE_FROM_ABI
  constexpr operator in_out_result<_I2, _O2>() && {
    return {std::move(in), std::move(out)};
  }
};

} // namespace ranges

#endif // _LIBCPP_STD_VER > 17 && !defined(_LIBCPP_HAS_NO_INCOMPLETE_RANGES)

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___ALGORITHM_IN_OUT_RESULT_H
