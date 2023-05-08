//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___ALGORITHM_PSTL_FILL_H
#define _LIBCPP___ALGORITHM_PSTL_FILL_H

#include <__algorithm/fill.h>
#include <__config>
#include <__iterator/iterator_traits.h>
#include <__pstl/internal/execution_impl.h>
#include <__pstl/internal/parallel_backend.h>
#include <__pstl/internal/parallel_backend_serial.h>
#include <__pstl/internal/parallel_impl.h>
#include <__pstl/internal/unseq_backend_simd.h>
#include <__type_traits/enable_if.h>
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
          class _Tp,
          enable_if_t<is_execution_policy_v<__remove_cvref_t<_ExecutionPolicy>>, int> = 0>
_LIBCPP_HIDE_FROM_ABI void
fill(_ExecutionPolicy&& __policy, _ForwardIterator __first, _ForwardIterator __last, const _Tp& __value) {
  if constexpr (__is_parallel_execution_policy_v<_ExecutionPolicy> &&
                __is_cpp17_random_access_iterator<_ForwardIterator>::value) {
    std::__terminate_on_exception([&] {
      __pstl::__par_backend::__parallel_for(
          __pstl::__internal::__par_backend_tag{},
          __policy,
          __first,
          __last,
          [&__policy, &__value](_ForwardIterator __brick_first, _ForwardIterator __brick_last) {
            std::fill(std::__remove_parallel_policy(__policy), __brick_first, __brick_last, __value);
          });
    });
  } else if constexpr (__is_unsequenced_execution_policy_v<_ExecutionPolicy> &&
                       __is_cpp17_random_access_iterator<_ForwardIterator>::value) {
    __pstl::__unseq_backend::__simd_fill_n(__first, __last - __first, __value);
  } else {
    std::fill(__first, __last, __value);
  }
}

template <class _ExecutionPolicy,
          class _ForwardIterator,
          class _SizeT,
          class _Tp,
          enable_if_t<is_execution_policy_v<__remove_cvref_t<_ExecutionPolicy>>, int> = 0>
_LIBCPP_HIDE_FROM_ABI void
fill_n(_ExecutionPolicy&& __policy, _ForwardIterator __first, _SizeT __n, const _Tp& __value) {
  if constexpr (__is_cpp17_random_access_iterator<_ForwardIterator>::value)
    std::fill(__policy, __first, __first + __n, __value);
  else
    std::fill_n(__first, __n, __value);
}

_LIBCPP_END_NAMESPACE_STD

#endif // !defined(_LIBCPP_HAS_NO_INCOMPLETE_PSTL) && _LIBCPP_STD_VER >= 17

#endif // _LIBCPP___ALGORITHM_PSTL_FILL_H
