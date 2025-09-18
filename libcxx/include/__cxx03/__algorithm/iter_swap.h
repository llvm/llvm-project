//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___CXX03___ALGORITHM_ITER_SWAP_H
#define _LIBCPP___CXX03___ALGORITHM_ITER_SWAP_H

#include <__cxx03/__config>
#include <__cxx03/__utility/declval.h>
#include <__cxx03/__utility/swap.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

template <class _ForwardIterator1, class _ForwardIterator2>
inline _LIBCPP_HIDE_FROM_ABI void iter_swap(_ForwardIterator1 __a, _ForwardIterator2 __b) {
  swap(*__a, *__b);
}

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___CXX03___ALGORITHM_ITER_SWAP_H
