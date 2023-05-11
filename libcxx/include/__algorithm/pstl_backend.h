//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___ALGORITHM_PSTL_BACKEND_H
#define _LIBCPP___ALGORITHM_PSTL_BACKEND_H

#include <__algorithm/pstl_backends/cpu_backend.h>
#include <__config>
#include <execution>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

#if !defined(_LIBCPP_HAS_NO_INCOMPLETE_PSTL) && _LIBCPP_STD_VER >= 17

_LIBCPP_BEGIN_NAMESPACE_STD

/*
TODO: Documentation of how backends work

A PSTL parallel backend is a tag type to which the following functions are associated, at minimum:

  template <class _ExecutionPolicy, class _Iterator, class _Func>
  void __pstl_for_each(_Backend, _ExecutionPolicy&&, _Iterator __first, _Iterator __last, _Func __f);

// TODO: Complete this list

The following functions are optional but can be provided. If provided, they are used by the corresponding
algorithms, otherwise they are implemented in terms of other algorithms. If none of the optional algorithms are
implemented, all the algorithms will eventually forward to the basis algorithms listed above:

  template <class _ExecutionPolicy, class _Iterator, class _Size, class _Func>
  void __pstl_for_each_n(_Backend, _ExecutionPolicy&&, _Iterator __first, _Size __n, _Func __f);

// TODO: Complete this list

*/

template <class _ExecutionPolicy>
struct __select_backend;

template <>
struct __select_backend<std::execution::sequenced_policy> {
  using type = __cpu_backend_tag;
};

#  if _LIBCPP_STD_VER >= 20
template <>
struct __select_backend<std::execution::unsequenced_policy> {
  using type = __cpu_backend_tag;
};
#  endif

#  if defined(_PSTL_CPU_BACKEND_SERIAL)
template <>
struct __select_backend<std::execution::parallel_policy> {
  using type = __cpu_backend_tag;
};

template <>
struct __select_backend<std::execution::parallel_unsequenced_policy> {
  using type = __cpu_backend_tag;
};

#  else

// ...New vendors can add parallel backends here...

#    error "Invalid choice of a PSTL parallel backend"
#  endif

_LIBCPP_END_NAMESPACE_STD

#endif // !defined(_LIBCPP_HAS_NO_INCOMPLETE_PSTL) && _LIBCPP_STD_VER >= 17

#endif // _LIBCPP___ALGORITHM_PSTL_BACKEND_H
