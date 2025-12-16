//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___ALGORITHM_GENERATE_N_H
#define _LIBCPP___ALGORITHM_GENERATE_N_H

#include <__algorithm/for_each_n.h>
#include <__config>
#include <__functional/identity.h>
#include <__utility/forward.h>
#include <__utility/move.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_PUSH_MACROS
#include <__undef_macros>

_LIBCPP_BEGIN_NAMESPACE_STD

template <class _OutputIterator, class _Size, class _Generator>
inline _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX20 _OutputIterator
__generate_n(_OutputIterator __first, _Size __orig_n, _Generator& __gen) {
  using __iter_ref = decltype(*__first);
  __identity __proj;
  auto __f = [&](__iter_ref __element) { std::forward<__iter_ref>(__element) = __gen(); };
  return std::__for_each_n(std::move(__first), __orig_n, __f, __proj);
}

template <class _OutputIterator, class _Size, class _Generator>
inline _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX20 _OutputIterator
generate_n(_OutputIterator __first, _Size __orig_n, _Generator __gen) {
  return std::__generate_n(std::move(__first), __orig_n, __gen);
}

_LIBCPP_END_NAMESPACE_STD

_LIBCPP_POP_MACROS

#endif // _LIBCPP___ALGORITHM_GENERATE_N_H
