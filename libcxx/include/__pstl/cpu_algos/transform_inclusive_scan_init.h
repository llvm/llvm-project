//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___PSTL_CPU_ALGOS_TRANSFORM_INCLUSIVE_SCAN_INIT_H
#define _LIBCPP___PSTL_CPU_ALGOS_TRANSFORM_INCLUSIVE_SCAN_INIT_H

#include <__config>
#include <__functional/identity.h>
#include <__functional/operations.h>
#include <__iterator/concepts.h>
#include <__iterator/iterator_traits.h>
#include <__numeric/transform_inclusive_scan.h>
#include <__numeric/transform_reduce.h>
#include <__optional/nullopt_t.h>
#include <__optional/optional.h>
#include <__pstl/backend_fwd.h>
#include <__pstl/cpu_algos/cpu_traits.h>
#include <__pstl/decoupled_lookback.h>
#include <__type_traits/is_execution_policy.h>
#include <__utility/move.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_PUSH_MACROS
#include <__undef_macros>

#if _LIBCPP_STD_VER >= 20 // TODO: should be 17 once https://github.com/llvm/llvm-project/pull/224356 is merged

_LIBCPP_BEGIN_NAMESPACE_STD
namespace __pstl {

template <class _Backend, class _RawExecutionPolicy>
struct __cpu_parallel_transform_inclusive_scan_init {
  template <class _Policy,
            class _ForwardIterator1,
            class _ForwardIterator2,
            class _BinaryOperation,
            class _UnaryOperation,
            class _Tp>
  _LIBCPP_HIDE_FROM_ABI optional<_ForwardIterator2> operator()(
      _Policy&&,
      _ForwardIterator1 __first,
      _ForwardIterator1 __last,
      _ForwardIterator2 __result,
      _BinaryOperation __reduce,
      _UnaryOperation __transform,
      _Tp __init) const noexcept {
    if constexpr (__is_parallel_execution_policy_v<_RawExecutionPolicy> &&
                  __has_random_access_iterator_category_or_concept<_ForwardIterator1>::value &&
                  __has_random_access_iterator_category_or_concept<_ForwardIterator2>::value) {
      auto __scan_head =
          [&](__empty /*__worker_ctx*/,
              _ForwardIterator1 __chunk_first,
              _ForwardIterator1 __chunk_last,
              __decoupled_lookback_partition<_Tp>* __optional_lookback_partition) {
            // Handling of the first chunk.
            // It is special because it doesn't have a previous partition and must immediately push its inclusive prefix
            // once the local reduction is complete.
            if (__optional_lookback_partition != nullptr) {
              // Calculate and publish the inclusive prefix only if the storage for this partition exists.
              // For the edge case when the entire workset consists of a single partition, the lookback will be empty.
              __optional_lookback_partition->__construct_inclusive_prefix(
                  std::transform_reduce(__chunk_first, __chunk_last, __init, __reduce, __transform));
            }

            // Derive the destination iterator for this chunk.
            _ForwardIterator2 __chunk_result = __result + (__chunk_first - __first);

            // Perform the scanning into the destination.
            std::transform_inclusive_scan(
                __chunk_first, __chunk_last, __chunk_result, __reduce, __transform, std::move(__init));
          };

      auto __scan_middle = [&](__empty /*__worker_ctx*/,
                               _ForwardIterator1 __chunk_first,
                               _ForwardIterator1 __chunk_last,
                               __decoupled_lookback_partition<_Tp>* __lookback_partition) {
        // General case: a chunk in the middle.

        // Compute and publish the local aggregate.
        _Tp __aggregate =
            std::transform_reduce(__chunk_first + 1, __chunk_last, __transform(*__chunk_first), __reduce, __transform);
        // __decoupled_lookback_partition< _Tp >& __partition = __lookback.__partition(__chunk_index);
        __lookback_partition->__construct_aggregate(__aggregate);

        // Derive the destination iterator for this chunk.
        _ForwardIterator2 __chunk_result = __result + (__chunk_first - __first);

        // Depending on the state of the previous chunk, either immediately compute and publish the inclusive prefix, or
        // first compute the inclusive prefix of the previous partition by ourselves and then compute and publish our
        // inclusive prefix.
        __decoupled_lookback_partition< _Tp >* __prev_partition = __lookback_partition - 1;
        if (__prev_partition->__acquire_available_status() & __decoupled_lookback_status_prefix_available) {
          // Use the existing inclusive prefix directly (avoid copies)
          const _Tp& __exclusive_prefix = __prev_partition->__inclusive_prefix();
          // Calculate and publish the inclusive prefix for the current chunk.
          __lookback_partition->__construct_inclusive_prefix(__reduce(__exclusive_prefix, std::move(__aggregate)));
          // Perform the scanning into the destination.
          std::transform_inclusive_scan(
              __chunk_first, __chunk_last, __chunk_result, __reduce, __transform, __exclusive_prefix);
        } else {
          // Calculate the exclusive prefix starting with the aggregate of the previous chunk.
          _Tp __exclusive_prefix = __calculate_inclusive_prefix_of_partition(__prev_partition, __reduce);
          // Calculate and publish the inclusive prefix for the current chunk.
          __lookback_partition->__construct_inclusive_prefix(__reduce(__exclusive_prefix, std::move(__aggregate)));
          // Perform the scanning into the destination.
          std::transform_inclusive_scan(
              __chunk_first, __chunk_last, __chunk_result, __reduce, __transform, std::move(__exclusive_prefix));
        }
      };

      auto __scan_tail =
          [&](__empty /*__worker_ctx*/,
              _ForwardIterator1 __chunk_first,
              _ForwardIterator1 __chunk_last,
              __decoupled_lookback_partition<_Tp>* __nonexistent_lookback_partition) {
            // Handling of the last chunk (which is also not the first one).
            // It doesn't have its lookback partition, so get the exclusive prefix from the previous chunk and perform
            // scanning without publishing the inclusive prefix.

            // Derive the destination iterator for this chunk.
            _ForwardIterator2 __chunk_result = __result + (__chunk_first - __first);

            __decoupled_lookback_partition< _Tp >* __prev_partition = __nonexistent_lookback_partition - 1;
            if (__prev_partition->__acquire_available_status() & __decoupled_lookback_status_prefix_available) {
              //  Perform the scanning using the existing inclusive prefix of the previous chunk.
              std::transform_inclusive_scan(
                  __chunk_first,
                  __chunk_last,
                  __chunk_result,
                  __reduce,
                  __transform,
                  __prev_partition->__inclusive_prefix());
            } else {
              // Calculate the exclusive prefix for the previous chunk and perform the scanning.
              std::transform_inclusive_scan(
                  __chunk_first,
                  __chunk_last,
                  __chunk_result,
                  __reduce,
                  __transform,
                  __calculate_inclusive_prefix_of_partition(__prev_partition, __reduce));
            }
          };

      auto __ret = __cpu_traits<_Backend>::template __lookback_scan<_Tp>(
          __first,
          __last,
          [](size_t /*__max_chunk_size*/) { return __empty{}; },
          __scan_head,
          __scan_middle,
          __scan_tail,
          [](__empty) {});
      if (!__ret)
        return nullopt;
      return __result + (__last - __first);
    } else {
      // Non-random access iterators cannot be processed in parallel, fall back to the sequential implementation.
      return std::transform_inclusive_scan(
          std::move(__first),
          std::move(__last),
          std::move(__result),
          std::move(__reduce),
          std::move(__transform),
          std::move(__init));
    }
  }
};

} // namespace __pstl
_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP_STD_VER >= 20

_LIBCPP_POP_MACROS

#endif // _LIBCPP___PSTL_CPU_ALGOS_TRANSFORM_INCLUSIVE_SCAN_INIT_H
