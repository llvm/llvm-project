//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___PSTL_CPU_ALGOS_UNINITIALIZED_ALGORITHMS_H
#define _LIBCPP___PSTL_CPU_ALGOS_UNINITIALIZED_ALGORITHMS_H

#include <__config>
#include <__iterator/concepts.h>
#include <__iterator/iterator_traits.h>
#include <__memory/uninitialized_algorithms.h>
#include <__optional/nullopt_t.h>
#include <__optional/optional.h>
#include <__pstl/backend_fwd.h>
#include <__pstl/cpu_algos/cpu_traits.h>
#include <__pstl/cpu_algos/for_each.h>
#include <__type_traits/is_execution_policy.h>
#include <__utility/move.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_PUSH_MACROS
#include <__undef_macros>

#if _LIBCPP_STD_VER >= 17

_LIBCPP_BEGIN_NAMESPACE_STD

namespace __pstl {

template <class _Backend, class _RawExecutionPolicy>
struct __cpu_parallel_uninitialized_copy {
  template <class _Policy, class _InputIterator, class _ForwardIterator>
  _LIBCPP_HIDE_FROM_ABI optional<_ForwardIterator>
  operator()(_Policy&&, _InputIterator __first, _InputIterator __last, _ForwardIterator __result) const noexcept {
    if constexpr (__is_parallel_execution_policy_v<_RawExecutionPolicy> &&
                  __has_random_access_iterator_category_or_concept<_InputIterator>::value &&
                  __has_random_access_iterator_category_or_concept<_ForwardIterator>::value) {
      auto __res = __parallel_for_each_iter_pair<_Backend>(
          __first,
          __last,
          __result,
          [](_InputIterator __brick_first, _InputIterator __brick_last, _ForwardIterator __brick_result) {
            std::uninitialized_copy(std::move(__brick_first), std::move(__brick_last), std::move(__brick_result));
          });
      if (!__res) {
        return std::nullopt; // Failed to run the algorithm, propagate the error.
      }
      return __result + (__last - __first);
    } else {
      // Non-random access iterators cannot be processed in parallel, fall back to the sequential implementation.
      return std::uninitialized_copy(std::move(__first), std::move(__last), std::move(__result));
    }
  }
};

template <class _Backend, class _RawExecutionPolicy>
struct __cpu_parallel_uninitialized_move {
  template <class _Policy, class _InputIterator, class _ForwardIterator>
  _LIBCPP_HIDE_FROM_ABI optional<_ForwardIterator>
  operator()(_Policy&&, _InputIterator __first, _InputIterator __last, _ForwardIterator __result) const noexcept {
    if constexpr (__is_parallel_execution_policy_v<_RawExecutionPolicy> &&
                  __has_random_access_iterator_category_or_concept<_InputIterator>::value &&
                  __has_random_access_iterator_category_or_concept<_ForwardIterator>::value) {
      auto __res = __parallel_for_each_iter_pair<_Backend>(
          __first,
          __last,
          __result,
          [](_InputIterator __brick_first, _InputIterator __brick_last, _ForwardIterator __brick_result) {
            std::uninitialized_move(std::move(__brick_first), std::move(__brick_last), std::move(__brick_result));
          });
      if (!__res) {
        return std::nullopt; // Failed to run the algorithm, propagate the error.
      }
      return __result + (__last - __first);
    } else {
      // Non-random access iterators cannot be processed in parallel, fall back to the sequential implementation.
      return std::uninitialized_move(std::move(__first), std::move(__last), std::move(__result));
    }
  }
};

} // namespace __pstl

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP_STD_VER >= 17

_LIBCPP_POP_MACROS

#endif // _LIBCPP___PSTL_CPU_ALGOS_UNINITIALIZED_ALGORITHMS_H
