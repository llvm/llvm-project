//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___ALGORITHM_PSTL_FOR_EACH_H
#define _LIBCPP___ALGORITHM_PSTL_FOR_EACH_H

#include <__algorithm/for_each.h>
#include <__algorithm/for_each_n.h>
#include <__config>
#include <__iterator/iterator_traits.h>
#include <__pstl/internal/parallel_backend.h>
#include <__pstl/internal/unseq_backend_simd.h>
#include <__type_traits/is_execution_policy.h>
#include <__type_traits/remove_cvref.h>
#include <__utility/terminate_on_exception.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

#if !defined(_LIBCPP_HAS_NO_INCOMPLETE_PSTL) && _LIBCPP_STD_VER >= 17

_LIBCPP_BEGIN_NAMESPACE_STD

template <class _ExecutionPolicy,
          class _ForwardIterator,
          class _Function,
          enable_if_t<is_execution_policy_v<__remove_cvref_t<_ExecutionPolicy>>, int> = 0>
_LIBCPP_HIDE_FROM_ABI void
for_each(_ExecutionPolicy&& __policy, _ForwardIterator __first, _ForwardIterator __last, _Function __func) {
  if constexpr (__is_parallel_execution_policy_v<_ExecutionPolicy> &&
                __is_cpp17_random_access_iterator<_ForwardIterator>::value) {
    std::__terminate_on_exception([&] {
      __pstl::__par_backend::__parallel_for(
          {},
          __policy,
          __first,
          __last,
          [&__policy, __func](_ForwardIterator __brick_first, _ForwardIterator __brick_last) {
            std::for_each(std::__remove_parallel_policy(__policy), __brick_first, __brick_last, __func);
          });
    });
  } else if constexpr (__is_unsequenced_execution_policy_v<_ExecutionPolicy> &&
                       __is_cpp17_random_access_iterator<_ForwardIterator>::value) {
    __pstl::__unseq_backend::__simd_walk_1(__first, __last - __first, __func);
  } else {
    std::for_each(__first, __last, __func);
  }
}

template <class _ExecutionPolicy,
          class _ForwardIterator,
          class _Size,
          class _Function,
          enable_if_t<is_execution_policy_v<__remove_cvref_t<_ExecutionPolicy>>, int> = 0>
_LIBCPP_HIDE_FROM_ABI void
for_each_n(_ExecutionPolicy&& __policy, _ForwardIterator __first, _Size __size, _Function __func) {
  if constexpr (__is_cpp17_random_access_iterator<_ForwardIterator>::value) {
    std::for_each(__policy, __first, __first + __size, __func);
  } else {
    std::for_each_n(__first, __size, __func);
  }
}

_LIBCPP_END_NAMESPACE_STD

#endif // !defined(_LIBCPP_HAS_NO_INCOMPLETE_PSTL) && _LIBCPP_STD_VER >= 17

#endif // _LIBCPP___ALGORITHM_PSTL_FOR_EACH_H
