// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___ALGORITHM_IN_FUN_RESULT_H
#define _LIBCPP___ALGORITHM_IN_FUN_RESULT_H

#include <__concepts/convertible_to.h>
#include <__config>
#include <__utility/move.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

#ifndef _LIBCPP_HAS_NO_CONCEPTS

namespace ranges {
template <class _Ip, class _Fp>
struct in_fun_result {
  _LIBCPP_NO_UNIQUE_ADDRESS _Ip in;
  _LIBCPP_NO_UNIQUE_ADDRESS _Fp fun;

  template <class _I2, class _F2>
    requires convertible_to<const _Ip&, _I2> && convertible_to<const _Fp&, _F2>
  _LIBCPP_HIDE_FROM_ABI constexpr operator in_fun_result<_I2, _F2>() const & {
    return {in, fun};
  }

  template <class _I2, class _F2>
    requires convertible_to<_Ip, _I2> && convertible_to<_Fp, _F2>
  _LIBCPP_HIDE_FROM_ABI constexpr operator in_fun_result<_I2, _F2>() && {
    return {std::move(in), std::move(fun)};
  }
};
} // namespace ranges

#endif // _LIBCPP_HAS_NO_RANGES

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___ALGORITHM_IN_FUN_RESULT_H
