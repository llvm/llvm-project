//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___PSTL_BACKENDS_STATPART_PARTITION_COPY_H
#define _LIBCPP___PSTL_BACKENDS_STATPART_PARTITION_COPY_H

#include <__algorithm/copy.h>
#include <__algorithm/partition_copy.h>
#include <__config>
#include <__cstddef/size_t.h>
#include <__execution/is_execution_policy.h>
#include <__functional/operations.h>
#include <__iterator/concepts.h>
#include <__iterator/iterator_traits.h>
#include <__optional/nullopt_t.h>
#include <__optional/optional.h>
#include <__pstl/backend_fwd.h>
#include <__pstl/backends/statpart/basis.h>
#include <__pstl/backends/statpart/bitset.h>
#include <__pstl/backends/statpart/copy_if.h>
#include <__pstl/backends/statpart/decoupled_lookback.h>
#include <__utility/move.h>
#include <__utility/pair.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_PUSH_MACROS
#include <__undef_macros>

#if _LIBCPP_STD_VER >= 17

_LIBCPP_BEGIN_NAMESPACE_STD
namespace __pstl {

namespace __statpart {

// __occupancy_pair carries the number of times the predicate returned 'true' and 'false'.
// A reduction can be performed over a range of occupancy pairs using its operator+.
struct __occupancy_pair {
  size_t __true_count;
  size_t __false_count;
  friend constexpr __occupancy_pair operator+(const __occupancy_pair& __lhs, const __occupancy_pair& __rhs) noexcept {
    return {__lhs.__true_count + __rhs.__true_count, __lhs.__false_count + __rhs.__false_count};
  }
};

// Copies the elements from [__first, __last) into [__out_true, ...) and [__out_false, ...),
// depending on the flag values in the bitset.
// If all elements go to the same output, the bitset lookup is skipped entirely. Returns the end iterators
// of the destination ranges.
template <class _ForwardIterator, class _ForwardOutIterator1, class _ForwardOutIterator2>
pair<_ForwardOutIterator1, _ForwardOutIterator2> __partition_copy_if_flag_set(
    _ForwardIterator __first,
    _ForwardIterator __last,
    _ForwardOutIterator1 __out_true,
    _ForwardOutIterator2 __out_false,
    __bitset_view __bitset,
    __occupancy_pair __occupancy) {
  if (__occupancy.__true_count == 0)
    return {__out_true, std::copy(__first, __last, __out_false)}; // Edge case: all elements go into `__out_false`.
  if (__occupancy.__false_count == 0)
    return {std::copy(__first, __last, __out_true), __out_false}; // Edge case: all elements go into `__out_true`.
  for (size_t __local_idx = 0; __first != __last; (void)++__first, ++__local_idx) {
    if (__bitset.__test(__local_idx)) {
      *__out_true = *__first;
      ++__out_true;
    } else {
      *__out_false = *__first;
      ++__out_false;
    }
  }
  return {__out_true, __out_false};
}

} // namespace __statpart

template <class _ExecutionPolicy>
struct __partition_copy<__statpart_backend_tag, _ExecutionPolicy> {
  template <class _Policy,
            class _ForwardIterator,
            class _ForwardOutIterator1,
            class _ForwardOutIterator2,
            class _Predicate>
  optional<pair<_ForwardOutIterator1, _ForwardOutIterator2>>
  operator()(_Policy&&,
             _ForwardIterator __first,
             _ForwardIterator __last,
             _ForwardOutIterator1 __out_true,
             _ForwardOutIterator2 __out_false,
             _Predicate __pred) const noexcept {
    if constexpr (__is_parallel_execution_policy_v<_ExecutionPolicy> &&
                  __has_random_access_iterator_category_or_concept<_ForwardIterator>::value &&
                  __has_random_access_iterator_category_or_concept<_ForwardOutIterator1>::value &&
                  __has_random_access_iterator_category_or_concept<_ForwardOutIterator2>::value) {
      // Each workers needs scratch memory to keep the bit flags to avoid calling the user predicate twice per element,
      // thus the setup callback allocates a dynamic bitset with enough storage for every worker.
      auto __setup = [](size_t __max_workers_count, size_t __max_partition_size) {
        return __statpart::__make_sharded_dynamic_bitset(__max_workers_count, __max_partition_size);
      };

      pair<_ForwardOutIterator1, _ForwardOutIterator2> __out_iters = {__out_true, __out_false};

      auto __scan_head =
          [&](__statpart::__sharded_dynamic_bitset& __ctx,
              size_t __worker_idx,
              _ForwardIterator __chunk_first,
              _ForwardIterator __chunk_last,
              __statpart::__decoupled_lookback_partition<__statpart::__occupancy_pair>* __optional_lookback_partition) {
            // Acquire the local scratch memory.
            __statpart::__bitset_view __bitset = __ctx.__shard(__worker_idx);

            // Populate the bitset for each of the elements in the chunk and get the occupancy.
            size_t __occupied_count =
                __statpart::__count_and_cache_predicate_results(__chunk_first, __chunk_last, __pred, __bitset);
            __statpart::__occupancy_pair __occupancy = {
                __occupied_count, static_cast<size_t>(__chunk_last - __chunk_first) - __occupied_count};

            // If we're not the only chunk - publish the final occupancy.
            if (__optional_lookback_partition != nullptr) {
              __optional_lookback_partition->__construct_inclusive_prefix(__occupancy);
            }

            // Copy the elements depending on the bitset flags.
            pair<_ForwardOutIterator1, _ForwardOutIterator2> __chunk_result = __statpart::__partition_copy_if_flag_set(
                __chunk_first, __chunk_last, __out_true, __out_false, __bitset, __occupancy);

            // If we're the last chunk - also report the positions of the last iterators in the result ranges.
            if (__optional_lookback_partition == nullptr) {
              __out_iters = __chunk_result;
            }
          };

      auto __scan_middle =
          [&](__statpart::__sharded_dynamic_bitset& __ctx,
              size_t __worker_idx,
              _ForwardIterator __chunk_first,
              _ForwardIterator __chunk_last,
              __statpart::__decoupled_lookback_partition<__statpart::__occupancy_pair>* __lookback_partition) {
            // Acquire the local scratch memory.
            __statpart::__bitset_view __bitset = __ctx.__shard(__worker_idx);

            // Populate the bitset for each of the elements in the chunk and get the occupancy.
            size_t __occupied_count =
                __statpart::__count_and_cache_predicate_results(__chunk_first, __chunk_last, __pred, __bitset);
            __statpart::__occupancy_pair __occupancy = {
                __occupied_count, static_cast<size_t>(__chunk_last - __chunk_first) - __occupied_count};

            // Publish the occupancy as a local aggregate first.
            __lookback_partition->__construct_aggregate(__occupancy);

            // Obtain the exclusive prefix.
            __statpart::__decoupled_lookback_partition<__statpart::__occupancy_pair>* __prev_partition =
                __lookback_partition - 1;
            __statpart::__occupancy_pair __exclusive_prefix =
                (__prev_partition->__acquire_available_status() &
                 __statpart::__decoupled_lookback_status_prefix_available)
                    ? __prev_partition->__inclusive_prefix()
                    : __statpart::__calculate_inclusive_prefix_of_partition(__prev_partition, plus<>{});

            // Publish the final occupancy.
            __lookback_partition->__construct_inclusive_prefix(__exclusive_prefix + __occupancy);

            // Copy the elements depending on the bitset flags.
            __statpart::__partition_copy_if_flag_set(
                __chunk_first,
                __chunk_last,
                __out_true + __exclusive_prefix.__true_count,
                __out_false + __exclusive_prefix.__false_count,
                __bitset,
                __occupancy);
          };

      auto __scan_tail = [&](__statpart::__sharded_dynamic_bitset& __ctx,
                             size_t __worker_idx,
                             _ForwardIterator __chunk_first,
                             _ForwardIterator __chunk_last,
                             __statpart::__decoupled_lookback_partition<__statpart::__occupancy_pair>*
                                 __nonexistent_lookback_partition) {
        // Acquire the local scratch memory.
        __statpart::__bitset_view __bitset = __ctx.__shard(__worker_idx);

        // Populate the bitset for each of the elements in the chunk and get the occupancy.
        size_t __occupied_count =
            __statpart::__count_and_cache_predicate_results(__chunk_first, __chunk_last, __pred, __bitset);
        __statpart::__occupancy_pair __occupancy = {
            __occupied_count, static_cast<size_t>(__chunk_last - __chunk_first) - __occupied_count};

        // Obtain the exclusive prefix.
        __statpart::__decoupled_lookback_partition<__statpart::__occupancy_pair>* __prev_partition =
            __nonexistent_lookback_partition - 1;
        __statpart::__occupancy_pair __exclusive_prefix =
            (__prev_partition->__acquire_available_status() & __statpart::__decoupled_lookback_status_prefix_available)
                ? __prev_partition->__inclusive_prefix()
                : __statpart::__calculate_inclusive_prefix_of_partition(__prev_partition, plus<>{});

        // Copy the elements depending on the bitset flags.
        pair<_ForwardOutIterator1, _ForwardOutIterator2> __chunk_result = __statpart::__partition_copy_if_flag_set(
            __chunk_first,
            __chunk_last,
            __out_true + __exclusive_prefix.__true_count,
            __out_false + __exclusive_prefix.__false_count,
            __bitset,
            __occupancy);

        // Report the positions of the last iterators in the result ranges.
        __out_iters = __chunk_result;
      };

      // Perform the partition copy as a single-pass scan with a reduction over occupancy pairs.
      auto __ret = __statpart::__lookback_scan<__statpart::__occupancy_pair>(
          __first, __last, __setup, __scan_head, __scan_middle, __scan_tail);
      if (!__ret)
        return nullopt;
      return __out_iters;

    } else {
      return std::partition_copy(
          std::move(__first), std::move(__last), std::move(__out_true), std::move(__out_false), std::move(__pred));
    }
  }
};

} // namespace __pstl
_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP_STD_VER >= 17

_LIBCPP_POP_MACROS

#endif // _LIBCPP___PSTL_BACKENDS_STATPART_PARTITION_COPY_H
