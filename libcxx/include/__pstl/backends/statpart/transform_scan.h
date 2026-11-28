//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___PSTL_BACKENDS_STATPART_TRANSFORM_SCAN_H
#define _LIBCPP___PSTL_BACKENDS_STATPART_TRANSFORM_SCAN_H

#include <__assert>
#include <__config>
#include <__cstddef/size_t.h>
#include <__execution/is_execution_policy.h>
#include <__functional/operations.h>
#include <__iterator/concepts.h>
#include <__iterator/iterator_traits.h>
#include <__numeric/transform_exclusive_scan.h>
#include <__numeric/transform_inclusive_scan.h>
#include <__numeric/transform_reduce.h>
#include <__optional/nullopt_t.h>
#include <__optional/optional.h>
#include <__pstl/backend_fwd.h>
#include <__pstl/backends/statpart/basis.h>
#include <__pstl/backends/statpart/decoupled_lookback.h>
#include <__utility/empty.h>
#include <__utility/forward.h>
#include <__utility/ignore_argument.h>
#include <__utility/move.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_PUSH_MACROS
#include <__undef_macros>

#if _LIBCPP_STD_VER >= 17

_LIBCPP_BEGIN_NAMESPACE_STD
namespace __pstl {

namespace __statpart {

template <bool _IsInclusive,
          class _Policy,
          class _ForwardIterator1,
          class _ForwardIterator2,
          class _BinaryOperation,
          class _UnaryOperation,
          class _Tp>
_LIBCPP_HIDE_FROM_ABI optional<_ForwardIterator2> __transform_scan(
    _Policy&&,
    _ForwardIterator1 __first,
    _ForwardIterator1 __last,
    _ForwardIterator2 __result,
    _BinaryOperation __reduce,
    _UnaryOperation __transform,
    _Tp __init) noexcept {
  // Perform either an inclusive or an exclusive scan, depending on the flavour of __transform_scan.
  // The arguments follow transform_inclusive_scan's order, the exclusive branch reorders them implicitly.
  auto __leaf_scan =
      [](auto&& __leaf_first,
         auto&& __leaf_last,
         auto&& __leaf_result,
         auto&& __leaf_reduce,
         auto&& __leaf_transform,
         auto&& __leaf_init) {
        if constexpr (_IsInclusive)
          return std::transform_inclusive_scan(
              std::forward<decltype(__leaf_first)>(__leaf_first),
              std::forward<decltype(__leaf_last)>(__leaf_last),
              std::forward<decltype(__leaf_result)>(__leaf_result),
              std::forward<decltype(__leaf_reduce)>(__leaf_reduce),
              std::forward<decltype(__leaf_transform)>(__leaf_transform),
              std::forward<decltype(__leaf_init)>(__leaf_init));
        else
          return std::transform_exclusive_scan(
              std::forward<decltype(__leaf_first)>(__leaf_first),
              std::forward<decltype(__leaf_last)>(__leaf_last),
              std::forward<decltype(__leaf_result)>(__leaf_result),
              std::forward<decltype(__leaf_init)>(__leaf_init),
              std::forward<decltype(__leaf_reduce)>(__leaf_reduce),
              std::forward<decltype(__leaf_transform)>(__leaf_transform));
      };

  if constexpr (__is_parallel_execution_policy_v<_Policy> &&
                __has_random_access_iterator_category_or_concept<_ForwardIterator1>::value &&
                __has_random_access_iterator_category_or_concept<_ForwardIterator2>::value) {
    auto __scan_head =
        [&](__ignore_argument /*__ctx*/,
            __ignore_argument /*__worker_idx*/,
            _ForwardIterator1 __chunk_first,
            _ForwardIterator1 __chunk_last,
            __decoupled_lookback_partition<_Tp>* __optional_lookback_partition) {
          // Handling of the first chunk.
          // It is special because it doesn't have a previous partition and must immediately publish its inclusive
          // prefix once the local reduction is complete.
          if (__optional_lookback_partition != nullptr) {
            // Calculate and publish the inclusive prefix only if the storage for this partition exists.
            // For the edge case when the entire workset consists of a single partition, the lookback won't be used.
            __optional_lookback_partition->__construct_inclusive_prefix(
                std::transform_reduce(__chunk_first, __chunk_last, __init, __reduce, __transform));
          }
          // Derive the destination iterator for this chunk.
          _ForwardIterator2 __chunk_result = __result + (__chunk_first - __first);
          // Perform the scanning into the destination.
          __leaf_scan(__chunk_first, __chunk_last, __chunk_result, __reduce, __transform, std::move(__init));
        };

    auto __scan_middle = [&](__ignore_argument /*__ctx*/,
                             __ignore_argument /*__worker_idx*/,
                             _ForwardIterator1 __chunk_first,
                             _ForwardIterator1 __chunk_last,
                             __decoupled_lookback_partition<_Tp>* __lookback_partition) {
      // General case: a chunk in the middle.
      _LIBCPP_ASSERT_INTERNAL(__lookback_partition != nullptr, "middle chunks must have their lookback partitions");

      // Compute and publish the local aggregate.
      _Tp __aggregate = std::transform_reduce(
          __chunk_first + 1, __chunk_last, static_cast<_Tp>(__transform(*__chunk_first)), __reduce, __transform);
      __lookback_partition->__construct_aggregate(__aggregate);

      // Derive the destination iterator for this chunk.
      _ForwardIterator2 __chunk_result = __result + (__chunk_first - __first);

      // Depending on the state of the previous partition, either immediately compute and publish the inclusive prefix,
      // or first compute the inclusive prefix of the previous partition by ourselves and then compute and publish
      // our inclusive prefix.
      __decoupled_lookback_partition<_Tp>* __prev_partition = __lookback_partition - 1;
      if (__prev_partition->__acquire_available_status() & __decoupled_lookback_status_prefix_available) {
        // Use the existing inclusive prefix directly.
        const _Tp& __exclusive_prefix = __prev_partition->__inclusive_prefix();
        // Calculate and publish the inclusive prefix for the current chunk.
        __lookback_partition->__construct_inclusive_prefix(__reduce(__exclusive_prefix, std::move(__aggregate)));
        // Perform the scanning into the destination.
        __leaf_scan(__chunk_first, __chunk_last, __chunk_result, __reduce, __transform, __exclusive_prefix);
      } else {
        // Calculate the exclusive prefix starting with the aggregate of the previous chunk.
        _Tp __exclusive_prefix = __statpart::__calculate_inclusive_prefix_of_partition(__prev_partition, __reduce);
        // Calculate and publish the inclusive prefix for the current chunk.
        __lookback_partition->__construct_inclusive_prefix(__reduce(__exclusive_prefix, std::move(__aggregate)));
        // Perform the scanning into the destination.
        __leaf_scan(__chunk_first, __chunk_last, __chunk_result, __reduce, __transform, std::move(__exclusive_prefix));
      }
    };

    auto __scan_tail = [&](__ignore_argument /*__ctx*/,
                           __ignore_argument /*__worker_idx*/,
                           _ForwardIterator1 __chunk_first,
                           _ForwardIterator1 __chunk_last,
                           __decoupled_lookback_partition<_Tp>* __nonexistent_lookback_partition) {
      // Handling of the last chunk (which is also not the first one).
      // It doesn't have its lookback partition, so get the exclusive prefix from the previous chunk and perform
      // scanning without publishing the inclusive prefix.

      // Derive the destination iterator for this chunk.
      _ForwardIterator2 __chunk_result = __result + (__chunk_first - __first);

      __decoupled_lookback_partition<_Tp>* __prev_partition = __nonexistent_lookback_partition - 1;
      if (__prev_partition->__acquire_available_status() & __decoupled_lookback_status_prefix_available) {
        // Perform the scanning using the existing inclusive prefix of the previous chunk.
        __leaf_scan(
            __chunk_first, __chunk_last, __chunk_result, __reduce, __transform, __prev_partition->__inclusive_prefix());
      } else {
        // Calculate the inclusive prefix for the previous chunk and perform the scanning.
        __leaf_scan(__chunk_first,
                    __chunk_last,
                    __chunk_result,
                    __reduce,
                    __transform,
                    __statpart::__calculate_inclusive_prefix_of_partition(__prev_partition, __reduce));
      }
    };

    auto __setup = [](size_t /*__max_workers_count*/, size_t /*__max_partition_size*/) {
      return optional<__empty>(__empty{}); // This algorithm doesn't require a custom context.
    };

    auto __ret = __statpart::__lookback_scan<_Tp>(__first, __last, __setup, __scan_head, __scan_middle, __scan_tail);
    if (!__ret)
      return nullopt;
    return __result + (__last - __first);
  } else {
    // Non-random access iterators cannot be processed in parallel, fall back to the sequential implementation.
    return __leaf_scan(
        std::move(__first),
        std::move(__last),
        std::move(__result),
        std::move(__reduce),
        std::move(__transform),
        std::move(__init));
  }
}

} // namespace __statpart

template <class _ExecutionPolicy>
struct __transform_inclusive_scan_init<__statpart_backend_tag, _ExecutionPolicy> {
  template <class _Policy,
            class _ForwardIterator1,
            class _ForwardIterator2,
            class _BinaryOperation,
            class _UnaryOperation,
            class _Tp>
  _LIBCPP_HIDE_FROM_ABI optional<_ForwardIterator2> operator()(
      _Policy&& __policy,
      _ForwardIterator1 __first,
      _ForwardIterator1 __last,
      _ForwardIterator2 __result,
      _BinaryOperation __reduce,
      _UnaryOperation __transform,
      _Tp __init) const noexcept {
    return __statpart::__transform_scan</*inclusive=*/true>(
        __policy,
        std::move(__first),
        std::move(__last),
        std::move(__result),
        std::move(__reduce),
        std::move(__transform),
        std::move(__init));
  }
};

template <class _ExecutionPolicy>
struct __transform_exclusive_scan<__statpart_backend_tag, _ExecutionPolicy> {
  template <class _Policy,
            class _ForwardIterator1,
            class _ForwardIterator2,
            class _Tp,
            class _BinaryOperation,
            class _UnaryOperation>
  _LIBCPP_HIDE_FROM_ABI optional<_ForwardIterator2> operator()(
      _Policy&& __policy,
      _ForwardIterator1 __first,
      _ForwardIterator1 __last,
      _ForwardIterator2 __result,
      _Tp __init,
      _BinaryOperation __reduce,
      _UnaryOperation __transform) const noexcept {
    return __statpart::__transform_scan</*exclusive=*/false>(
        __policy,
        std::move(__first),
        std::move(__last),
        std::move(__result),
        std::move(__reduce),
        std::move(__transform),
        std::move(__init));
  }
};

} // namespace __pstl
_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP_STD_VER >= 17

_LIBCPP_POP_MACROS

#endif // _LIBCPP___PSTL_BACKENDS_STATPART_TRANSFORM_SCAN_H
