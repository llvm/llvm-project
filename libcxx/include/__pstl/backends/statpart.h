//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___PSTL_STATPART_H
#define _LIBCPP___PSTL_STATPART_H

#include <__pstl/backends/statpart/basis.h>
#include <__pstl/backends/statpart/find_end.h>
#include <__pstl/backends/statpart/find_if.h>
#include <__pstl/backends/statpart/for_each.h>
#include <__pstl/backends/statpart/is_heap_until.h>
#include <__pstl/backends/statpart/merge.h>
#include <__pstl/backends/statpart/min_element.h>
#include <__pstl/backends/statpart/minmax_element.h>
#include <__pstl/backends/statpart/mismatch.h>
#include <__pstl/backends/statpart/reverse.h>
#include <__pstl/backends/statpart/search.h>
#include <__pstl/backends/statpart/search_n.h>
#include <__pstl/backends/statpart/stable_sort.h>
#include <__pstl/backends/statpart/swap_ranges.h>
#include <__pstl/backends/statpart/transform.h>

#include <__algorithm/find_end.h>
#include <__algorithm/find_if.h>
#include <__algorithm/for_each.h>
#include <__algorithm/is_heap_until.h>
#include <__algorithm/lower_bound.h>
#include <__algorithm/merge.h>
#include <__algorithm/min.h>
#include <__algorithm/min_element.h>
#include <__algorithm/minmax_element.h>
#include <__algorithm/mismatch.h>
#include <__algorithm/move.h>
#include <__algorithm/reverse.h>
#include <__algorithm/search.h>
#include <__algorithm/search_n.h>
#include <__algorithm/stable_sort.h>
#include <__algorithm/swap_ranges.h>
#include <__algorithm/transform.h>
#include <__algorithm/upper_bound.h>
#include <__atomic/atomic.h>
#include <__config>
#include <__functional/identity.h>
#include <__functional/operations.h>
#include <__iterator/concepts.h>
#include <__iterator/iterator_traits.h>
#include <__iterator/move_iterator.h>
#include <__iterator/next.h>
#include <__iterator/reverse_iterator.h>
#include <__memory/allocator.h>
#include <__memory/construct_at.h>
#include <__memory/destroy.h>
#include <__memory/uninitialized_algorithms.h>
#include <__memory/unique_ptr.h>
#include <__numeric/reduce.h>
#include <__numeric/transform_reduce.h>
#include <__optional/nullopt_t.h>
#include <__optional/optional.h>
#include <__pstl/backend_fwd.h>
#include <__pstl/cpu_algos/cpu_traits.h>
#include <__pstl/cpu_algos/find_if.h>
#include <__thread/thread.h>
#include <__type_traits/is_assignable.h>
#include <__type_traits/is_constructible.h>
#include <__type_traits/is_execution_policy.h>
#include <__type_traits/is_trivially_copyable.h>
#include <__utility/convert_to_integral.h>
#include <__utility/empty.h>
#include <__utility/move.h>
#include <__utility/pair.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_PUSH_MACROS
#include <__undef_macros>

#if _LIBCPP_STD_VER >= 17

_LIBCPP_BEGIN_NAMESPACE_STD
_LIBCPP_BEGIN_EXPLICIT_ABI_ANNOTATIONS
namespace __pstl {

template <class _ExecutionPolicy>
struct __transform_reduce<__statpart_backend_tag, _ExecutionPolicy> {
  template <class _Policy, class _ForwardIterator, class _Tp, class _BinaryOperation, class _UnaryOperation>
  _LIBCPP_HIDE_FROM_ABI optional<_Tp>
  operator()(_Policy&&,
             _ForwardIterator __first,
             _ForwardIterator __last,
             _Tp __init,
             _BinaryOperation __reduce,
             _UnaryOperation __transform) const noexcept {
    if constexpr (__is_parallel_execution_policy_v<_ExecutionPolicy> &&
                  __has_random_access_iterator_category_or_concept<_ForwardIterator>::value) {
      return __statpart::__transform_reduce(
          std::move(__first),
          std::move(__last),
          [__transform](_ForwardIterator __iter) { return __transform(*__iter); },
          std::move(__init),
          __reduce,
          [__transform, __reduce](auto __brick_first, auto __brick_last, _Tp __brick_init) {
            return std::transform_reduce(
                std::move(__brick_first),
                std::move(__brick_last),
                std::move(__brick_init),
                std::move(__reduce),
                std::move(__transform));
          });
    } else {
      return std::transform_reduce(
          std::move(__first), std::move(__last), std::move(__init), std::move(__reduce), std::move(__transform));
    }
  }
};

template <class _ExecutionPolicy>
struct __transform_reduce_binary<__statpart_backend_tag, _ExecutionPolicy> {
  template <class _Policy,
            class _ForwardIterator1,
            class _ForwardIterator2,
            class _Tp,
            class _BinaryOperation1,
            class _BinaryOperation2>
  _LIBCPP_HIDE_FROM_ABI optional<_Tp> operator()(
      _Policy&&,
      _ForwardIterator1 __first1,
      _ForwardIterator1 __last1,
      _ForwardIterator2 __first2,
      _Tp __init,
      _BinaryOperation1 __reduce,
      _BinaryOperation2 __transform) const noexcept {
    if constexpr (__is_parallel_execution_policy_v<_ExecutionPolicy> &&
                  __has_random_access_iterator_category_or_concept<_ForwardIterator1>::value &&
                  __has_random_access_iterator_category_or_concept<_ForwardIterator2>::value) {
      return __statpart::__transform_reduce(
          __first1,
          std::move(__last1),
          [__first1, __first2, __transform](_ForwardIterator1 __iter) {
            return __transform(*__iter, *(__first2 + (__iter - __first1)));
          },
          std::move(__init),
          std::move(__reduce),
          [__first1, __first2, __reduce, __transform](
              _ForwardIterator1 __brick_first, _ForwardIterator1 __brick_last, _Tp __brick_init) {
            return std::transform_reduce(
                __brick_first,
                std::move(__brick_last),
                __first2 + (__brick_first - __first1),
                std::move(__brick_init),
                std::move(__reduce),
                std::move(__transform));
          });
    } else {
      return std::transform_reduce(
          std::move(__first1),
          std::move(__last1),
          std::move(__first2),
          std::move(__init),
          std::move(__reduce),
          std::move(__transform));
    }
  }
};

template <class _ExecutionPolicy>
struct __uninitialized_copy<__statpart_backend_tag, _ExecutionPolicy> {
  template <class _Policy, class _ForwardIterator1, class _ForwardIterator2>
  _LIBCPP_HIDE_FROM_ABI optional<_ForwardIterator2> operator()(
      _Policy&&, _ForwardIterator1 __first, _ForwardIterator1 __last, _ForwardIterator2 __result) const noexcept {
    if constexpr (__is_parallel_execution_policy_v<_ExecutionPolicy> &&
                  __has_random_access_iterator_category_or_concept<_ForwardIterator1>::value &&
                  __has_random_access_iterator_category_or_concept<_ForwardIterator2>::value) {
      auto __res = __statpart::__for_each_iter_pair(
          __first,
          __last,
          __result,
          [](_ForwardIterator1 __brick_first, _ForwardIterator1 __brick_last, _ForwardIterator2 __brick_result) {
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

template <class _ExecutionPolicy>
struct __uninitialized_move<__statpart_backend_tag, _ExecutionPolicy> {
  template <class _Policy, class _ForwardIterator1, class _ForwardIterator2>
  _LIBCPP_HIDE_FROM_ABI optional<_ForwardIterator2> operator()(
      _Policy&&, _ForwardIterator1 __first, _ForwardIterator1 __last, _ForwardIterator2 __result) const noexcept {
    if constexpr (__is_parallel_execution_policy_v<_ExecutionPolicy> &&
                  __has_random_access_iterator_category_or_concept<_ForwardIterator1>::value &&
                  __has_random_access_iterator_category_or_concept<_ForwardIterator2>::value) {
      auto __res = __statpart::__for_each_iter_pair(
          __first,
          __last,
          __result,
          [](_ForwardIterator1 __brick_first, _ForwardIterator1 __brick_last, _ForwardIterator2 __brick_result) {
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
_LIBCPP_END_EXPLICIT_ABI_ANNOTATIONS
_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP_STD_VER >= 17

_LIBCPP_POP_MACROS

#endif // _LIBCPP___PSTL_STATPART_H
