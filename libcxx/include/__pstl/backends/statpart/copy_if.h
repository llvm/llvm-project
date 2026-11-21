//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___PSTL_BACKENDS_STATPART_COPY_IF_H
#define _LIBCPP___PSTL_BACKENDS_STATPART_COPY_IF_H

#include <__algorithm/copy_if.h>
#include <__bit/popcount.h>
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
#include <__pstl/backends/statpart/decoupled_lookback.h>
#include <__utility/move.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_PUSH_MACROS
#include <__undef_macros>

#if _LIBCPP_STD_VER >= 17

_LIBCPP_BEGIN_NAMESPACE_STD
_LIBCPP_BEGIN_EXPLICIT_ABI_ANNOTATIONS
namespace __pstl {

namespace __statpart {

// Applies the predicate to each element in [first, last), stores the results in the bitset and returns the number of
// 'true' responses of the predicate. Bitset is not required to be initialized before calling this function.
template <class _ForwardIterator, class _Pred>
_LIBCPP_HIDE_FROM_ABI size_t __count_and_cache_predicate_results(
    _ForwardIterator __first, _ForwardIterator __last, _Pred& __pred, __bitset_view __bitset) {
  size_t __occupied_count = 0;
  size_t* __bits_word_ptr = __bitset.__bits_;
  size_t __bits_word      = 0;
  size_t __bit_idx        = 0;
  for (; __first != __last; ++__first) {
    bool __pred_val = static_cast<bool>(__pred(*__first));
    __bits_word |= static_cast<size_t>(__pred_val) << __bit_idx; // Update the current word with this predicate result
    if (++__bit_idx == __bitset_view::__bits_per_word_) {
      __occupied_count += std::__popcount(__bits_word); // Update the total occupancy with this word
      *__bits_word_ptr++ = __bits_word;                 // Store the collected flags
      __bits_word        = 0;
      __bit_idx          = 0;
    }
  }
  if (__bit_idx != 0) { // Process any leftovers in the current word
    __occupied_count += std::__popcount(__bits_word);
    *__bits_word_ptr = __bits_word;
  }
  return __occupied_count;
}

// Copies the elements from [first, last) into [result, ...) depending on the flag values in the bitset.
// Returns the end iterator of the destination range.
template <class _ForwardIterator1, class _ForwardIterator2>
_LIBCPP_HIDE_FROM_ABI _ForwardIterator2 __copy_if_flag_set(
    _ForwardIterator1 __first, _ForwardIterator1 __last, _ForwardIterator2 __result, __bitset_view __bitset) {
  size_t __local_idx = 0;
  for (auto __it = __first; __it != __last; ++__it) {
    if (__bitset.__test(__local_idx)) {
      *__result = *__it;
      ++__result;
    }
    ++__local_idx;
  }
  return __result;
}

} // namespace __statpart

template <class _ExecutionPolicy>
struct __copy_if<__statpart_backend_tag, _ExecutionPolicy> {
  template <class _Policy, class _ForwardIterator1, class _ForwardIterator2, class _Pred>
  _LIBCPP_HIDE_FROM_ABI optional<_ForwardIterator2>
  operator()(_Policy&&, _ForwardIterator1 __first, _ForwardIterator1 __last, _ForwardIterator2 __result, _Pred __pred)
      const noexcept {
    if constexpr (__is_parallel_execution_policy_v<_ExecutionPolicy> &&
                  __has_random_access_iterator_category_or_concept<_ForwardIterator1>::value &&
                  __has_random_access_iterator_category_or_concept<_ForwardIterator2>::value) {
      auto __setup = [](size_t __max_workers_count, size_t __max_partition_size) {
        return __statpart::__make_sharded_dynamic_bitset(__max_workers_count, __max_partition_size);
      };

      _ForwardIterator2 __out_iter = __result;

      auto __scan_head =
          [&](__statpart::__sharded_dynamic_bitset& __ctx,
              size_t __worker_idx,
              _ForwardIterator1 __chunk_first,
              _ForwardIterator1 __chunk_last,
              __statpart::__decoupled_lookback_partition<size_t>* __optional_lookback_partition) {
            // Acquire the local scratch memory
            __statpart::__bitset_view __bitset = __ctx.__shard(__worker_idx);

            // Populate the bitset for each of the elements in the chunk and get the occupancy
            size_t __occupied_count =
                __statpart::__count_and_cache_predicate_results(__chunk_first, __chunk_last, __pred, __bitset);

            // If we're not the only chunk - publish the final occupancy
            if (__optional_lookback_partition != nullptr) {
              __optional_lookback_partition->__construct_inclusive_prefix(__occupied_count);
            }

            // Copy the elements depending on the bitset flags
            _ForwardIterator2 __chunk_result =
                __statpart::__copy_if_flag_set(__chunk_first, __chunk_last, __result, __bitset);

            // If we're the last chunk - also the report the position of the last iterator in the result range
            if (__optional_lookback_partition == nullptr) {
              __out_iter = __chunk_result;
            }
          };

      auto __scan_middle = [&](__statpart::__sharded_dynamic_bitset& __ctx,
                               size_t __worker_idx,
                               _ForwardIterator1 __chunk_first,
                               _ForwardIterator1 __chunk_last,
                               __statpart::__decoupled_lookback_partition<size_t>* __lookback_partition) {
        // Acquire the local scratch memory
        __statpart::__bitset_view __bitset = __ctx.__shard(__worker_idx);

        // Populate the bitset for each of the elements in the chunk and get the occupancy
        size_t __occupied_count =
            __statpart::__count_and_cache_predicate_results(__chunk_first, __chunk_last, __pred, __bitset);

        // Publish the occupancy as a local aggregate first
        __lookback_partition->__construct_aggregate(__occupied_count);

        // Obtain the exclusive prefix
        __statpart::__decoupled_lookback_partition<size_t>* __prev_partition = __lookback_partition - 1;
        size_t __exclusive_prefix =
            (__prev_partition->__acquire_available_status() & __statpart::__decoupled_lookback_status_prefix_available)
                ? __prev_partition->__inclusive_prefix()
                : __statpart::__calculate_inclusive_prefix_of_partition(__prev_partition, plus<>{});

        // Publish the final occupancy
        __lookback_partition->__construct_inclusive_prefix(__exclusive_prefix + __occupied_count);

        // Copy the elements depending on the bitset flags
        __statpart::__copy_if_flag_set(__chunk_first, __chunk_last, __result + __exclusive_prefix, __bitset);
      };

      auto __scan_tail = [&](__statpart::__sharded_dynamic_bitset& __ctx,
                             size_t __worker_idx,
                             _ForwardIterator1 __chunk_first,
                             _ForwardIterator1 __chunk_last,
                             __statpart::__decoupled_lookback_partition<size_t>* __nonexistent_lookback_partition) {
        // Acquire the local scratch memory
        __statpart::__bitset_view __bitset = __ctx.__shard(__worker_idx);

        // Populate the bitset for each of the elements in the chunk and get the occupancy
        __statpart::__count_and_cache_predicate_results(__chunk_first, __chunk_last, __pred, __bitset);

        // Obtain the exclusive prefix
        __statpart::__decoupled_lookback_partition< size_t >* __prev_partition = __nonexistent_lookback_partition - 1;
        size_t __exclusive_prefix =
            (__prev_partition->__acquire_available_status() & __statpart::__decoupled_lookback_status_prefix_available)
                ? __prev_partition->__inclusive_prefix()
                : __statpart::__calculate_inclusive_prefix_of_partition(__prev_partition, plus<>{});

        // Copy the elements depending on the bitset flags
        _ForwardIterator2 __chunk_result =
            __statpart::__copy_if_flag_set(__chunk_first, __chunk_last, __result + __exclusive_prefix, __bitset);

        // Report the position of the last iterator in the result range
        __out_iter = __chunk_result;
      };

      auto __ret =
          __statpart::__lookback_scan<size_t>(__first, __last, __setup, __scan_head, __scan_middle, __scan_tail);
      if (!__ret)
        return nullopt;
      return __out_iter;
    } else {
      return std::copy_if(std::move(__first), std::move(__last), std::move(__result), std::move(__pred));
    }
  }
};

} // namespace __pstl
_LIBCPP_END_EXPLICIT_ABI_ANNOTATIONS
_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP_STD_VER >= 17

_LIBCPP_POP_MACROS

#endif // _LIBCPP___PSTL_BACKENDS_STATPART_COPY_IF_H
