//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___ALGORITHM_PSTL_FIND_H
#define _LIBCPP___ALGORITHM_PSTL_FIND_H

#include <__algorithm/comp.h>
#include <__algorithm/find.h>
#include <__config>
#include <__functional/operations.h>
#include <__iterator/iterator_traits.h>
#include <__pstl/internal/execution_impl.h>
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
_LIBCPP_HIDE_FROM_ABI _ForwardIterator
find(_ExecutionPolicy&& __policy, _ForwardIterator __first, _ForwardIterator __last, const _Tp& __value) {
  if constexpr (__is_parallel_execution_policy_v<_ExecutionPolicy> &&
                __is_cpp17_random_access_iterator<_ForwardIterator>::value) {
    return std::__terminate_on_exception([&] {
      return __pstl::__internal::__parallel_find(
          __pstl::__internal::__par_backend_tag{},
          __policy,
          __first,
          __last,
          [&__policy, &__value](_ForwardIterator __brick_first, _ForwardIterator __brick_last) {
            return std::find(std::__remove_parallel_policy(__policy), __brick_first, __brick_last, __value);
          },
          less<>{},
          true);
    });
  } else if constexpr (__is_unsequenced_execution_policy_v<_ExecutionPolicy> &&
                       __is_cpp17_random_access_iterator<_ForwardIterator>::value) {
    using __diff_t = __iter_diff_t<_ForwardIterator>;
    return __pstl::__unseq_backend::__simd_first(
        __first, __diff_t(0), __last - __first, [&__value](_ForwardIterator __iter, __diff_t __i) {
          return __iter[__i] == __value;
        });
  } else {
    return std::find(__first, __last, __value);
  }
}

template <class _ExecutionPolicy,
          class _ForwardIterator,
          class _Predicate,
          enable_if_t<is_execution_policy_v<__remove_cvref_t<_ExecutionPolicy>>, int> = 0>
_LIBCPP_HIDE_FROM_ABI _ForwardIterator
find_if(_ExecutionPolicy&& __policy, _ForwardIterator __first, _ForwardIterator __last, _Predicate __pred) {
  if constexpr (__is_parallel_execution_policy_v<_ExecutionPolicy> &&
                __is_cpp17_random_access_iterator<_ForwardIterator>::value) {
    return std::__terminate_on_exception([&] {
      return __pstl::__internal::__parallel_find(
          __pstl::__internal::__par_backend_tag{},
          __policy,
          __first,
          __last,
          [&__policy, &__pred](_ForwardIterator __brick_first, _ForwardIterator __brick_last) {
            return std::find_if(std::__remove_parallel_policy(__policy), __brick_first, __brick_last, __pred);
          },
          less<>{},
          true);
    });
  } else if constexpr (__is_unsequenced_execution_policy_v<_ExecutionPolicy> &&
                       __is_cpp17_random_access_iterator<_ForwardIterator>::value) {
    using __diff_t = __iter_diff_t<_ForwardIterator>;
    return __pstl::__unseq_backend::__simd_first(
        __first, __diff_t(0), __last - __first, [&__pred](_ForwardIterator __iter, __diff_t __i) {
          return __pred(__iter[__i]);
        });
  } else {
    return std::find_if(__first, __last, __pred);
  }
}

template <class _ExecutionPolicy,
          class _ForwardIterator,
          class _Predicate,
          enable_if_t<is_execution_policy_v<__remove_cvref_t<_ExecutionPolicy>>, int> = 0>
_LIBCPP_HIDE_FROM_ABI _ForwardIterator
find_if_not(_ExecutionPolicy&& __policy, _ForwardIterator __first, _ForwardIterator __last, _Predicate __pred) {
  return std::find_if(__policy, __first, __last, [&](__iter_reference<_ForwardIterator> __value) {
    return !__pred(__value);
  });
}

_LIBCPP_END_NAMESPACE_STD

#endif // !defined(_LIBCPP_HAS_NO_INCOMPLETE_PSTL) && _LIBCPP_STD_VER >= 17

#endif // _LIBCPP___ALGORITHM_PSTL_FIND_H
