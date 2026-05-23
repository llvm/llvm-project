//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___ALGORITHM_REPLACE_H
#define _LIBCPP___ALGORITHM_REPLACE_H

#include <__algorithm/replace_if.h>
#include <__config>
#include <__utility/move.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

template <class _ForwardIterator, class _Tp>
inline _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX20 void
replace(_ForwardIterator __first, _ForwardIterator __last, const _Tp& __old_value, const _Tp& __new_value) {
  auto __pred = [&__old_value](const auto& __curr) { return __curr == __old_value; };

  std::replace_if(std::move(__first), std::move(__last), __pred, __new_value);
}

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___ALGORITHM_REPLACE_H
