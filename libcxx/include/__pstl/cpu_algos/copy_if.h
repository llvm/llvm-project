//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___PSTL_CPU_ALGOS_COPY_IF_H
#define _LIBCPP___PSTL_CPU_ALGOS_COPY_IF_H

#include <__algorithm/copy_if.h>
#include <__algorithm/fill.h>
#include <__config>
#include <__cstddef/size_t.h>
#include <__functional/operations.h>
#include <__iterator/concepts.h>
#include <__iterator/iterator_traits.h>
#include <__memory/unique_ptr.h>
#include <__new/global_new_delete.h>
#include <__new/nothrow_t.h>
#include <__optional/nullopt_t.h>
#include <__optional/optional.h>
#include <__pstl/backend_fwd.h>
#include <__pstl/cpu_algos/cpu_traits.h>
#include <__pstl/cpu_algos/find_if.h>
#include <__pstl/decoupled_lookback.h>
#include <__type_traits/is_execution_policy.h>
#include <__utility/move.h>
#include <climits> // for CHAR_BIT

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_PUSH_MACROS
#include <__undef_macros>

#if _LIBCPP_STD_VER >= 20 // TODO: should be 17 once https://github.com/llvm/llvm-project/pull/224356 is merged

_LIBCPP_BEGIN_NAMESPACE_STD
namespace __pstl {

struct _LIBCPP_HIDE_FROM_ABI __dynamic_bitset {
  size_t __size_;
  std::unique_ptr<size_t[]> __bits_;

  _LIBCPP_HIDE_FROM_ABI __dynamic_bitset(size_t __max_chunk_size)
      : __size_((__max_chunk_size + CHAR_BIT * sizeof(size_t) - 1) / (CHAR_BIT * sizeof(size_t))),
        __bits_(new (std::nothrow) size_t[__size_]) {}
  _LIBCPP_HIDE_FROM_ABI __dynamic_bitset(__dynamic_bitset&&) noexcept            = default;
  _LIBCPP_HIDE_FROM_ABI __dynamic_bitset& operator=(__dynamic_bitset&&) noexcept = default;
  _LIBCPP_HIDE_FROM_ABI bool __test(size_t __pos) const {
    size_t __word_index = __pos / (CHAR_BIT * sizeof(size_t));
    size_t __bit_index  = __pos % (CHAR_BIT * sizeof(size_t));
    return (__bits_[__word_index] & (size_t(1) << __bit_index)) != 0;
  }
  _LIBCPP_HIDE_FROM_ABI void __set(size_t __pos) {
    size_t __word_index = __pos / (CHAR_BIT * sizeof(size_t));
    size_t __bit_index  = __pos % (CHAR_BIT * sizeof(size_t));
    __bits_[__word_index] |= (size_t(1) << __bit_index);
  }
  _LIBCPP_HIDE_FROM_ABI void __reset(size_t __pos) {
    size_t __word_index = __pos / (CHAR_BIT * sizeof(size_t));
    size_t __bit_index  = __pos % (CHAR_BIT * sizeof(size_t));
    __bits_[__word_index] &= ~(size_t(1) << __bit_index);
  }
  _LIBCPP_HIDE_FROM_ABI void __reset() {
    for (size_t __i = 0; __i < __size_; ++__i) {
      __bits_[__i] = 0;
    }
  }
};

template <class _ForwardIterator, class _Pred>
_LIBCPP_HIDE_FROM_ABI size_t __count_and_cache_predicate_results(
    _ForwardIterator __first, _ForwardIterator __last, const _Pred& __pred, __dynamic_bitset& __bitset) {
  size_t __occupied_count = 0;
  size_t __local_idx      = 0;
  for (auto __it = __first; __it != __last; ++__it) {
    if (__pred(*__it)) {
      __bitset.__set(__local_idx);
      ++__occupied_count;
    }
    ++__local_idx;
  }
  return __occupied_count;
}

template <class _ForwardIterator1, class _ForwardIterator2>
_LIBCPP_HIDE_FROM_ABI _ForwardIterator2 __copy_if_flag_set(
    _ForwardIterator1 __first, _ForwardIterator1 __last, _ForwardIterator2 __result, __dynamic_bitset& __bitset) {
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

template <class _Backend, class _RawExecutionPolicy>
struct __cpu_parallel_copy_if {
  template <class _Policy, class _ForwardIterator1, class _ForwardIterator2, class _Pred>
  _LIBCPP_HIDE_FROM_ABI optional<_ForwardIterator2>
  operator()(_Policy&&, _ForwardIterator1 __first, _ForwardIterator1 __last, _ForwardIterator2 __result, _Pred __pred)
      const noexcept {
    if constexpr (__is_parallel_execution_policy_v<_RawExecutionPolicy> &&
                  __has_random_access_iterator_category_or_concept<_ForwardIterator1>::value &&
                  __has_random_access_iterator_category_or_concept<_ForwardIterator2>::value) {
      auto __prologue = [](size_t __max_chunk_size) { return __dynamic_bitset(__max_chunk_size); };
      auto __epilogue = [](__dynamic_bitset&&) {};

      _ForwardIterator2 __out_iter = __result;

      // TODO: support the case when the worker context was not allocated due to malloc failure?

      auto __scan_head =
          [&](__dynamic_bitset& __worker_ctx,
              _ForwardIterator1 __chunk_first,
              _ForwardIterator1 __chunk_last,
              __decoupled_lookback_partition<size_t>* __optional_lookback_partition) {
            // Reset the local storage by setting all flags to false
            __worker_ctx.__reset();

            // Populate the bitset for each of the elements in the chunk and get the occupancy
            size_t __occupied_count =
                __pstl::__count_and_cache_predicate_results(__chunk_first, __chunk_last, __pred, __worker_ctx);

            // If we're not the only chunk - publish the final occupancy
            if (__optional_lookback_partition != nullptr) {
              __optional_lookback_partition->__construct_inclusive_prefix(__occupied_count);
            }

            // Copy the elements depending on the bitset flags
            _ForwardIterator2 __chunk_result =
                __pstl::__copy_if_flag_set(__chunk_first, __chunk_last, __result, __worker_ctx);

            // If we're the last chunk - also the report the position of the last iterator in the result range
            if (__optional_lookback_partition == nullptr) {
              __out_iter = __chunk_result;
            }
          };

      auto __scan_middle =
          [&](__dynamic_bitset& __worker_ctx,
              _ForwardIterator1 __chunk_first,
              _ForwardIterator1 __chunk_last,
              __decoupled_lookback_partition<size_t>* __lookback_partition) {
            // Reset the local storage by setting all flags to false
            __worker_ctx.__reset();

            // Populate the bitset for each of the elements in the chunk and get the occupancy
            size_t __occupied_count =
                __pstl::__count_and_cache_predicate_results(__chunk_first, __chunk_last, __pred, __worker_ctx);

            // Publish the occupancy as a local aggregate first
            __lookback_partition->__construct_aggregate(__occupied_count);

            // Obtain the exclusive prefix
            __decoupled_lookback_partition<size_t>* __prev_partition = __lookback_partition - 1;
            size_t __exclusive_prefix =
                (__prev_partition->__acquire_available_status() & __decoupled_lookback_status_prefix_available)
                    ? __prev_partition->__inclusive_prefix()
                    : __pstl::__calculate_inclusive_prefix_of_partition(__prev_partition, plus<>{});

            // Publish the final occupancy
            __lookback_partition->__construct_inclusive_prefix(__exclusive_prefix + __occupied_count);

            // Copy the elements depending on the bitset flags
            __pstl::__copy_if_flag_set(__chunk_first, __chunk_last, __result + __exclusive_prefix, __worker_ctx);
          };

      auto __scan_tail =
          [&](__dynamic_bitset& __worker_ctx,
              _ForwardIterator1 __chunk_first,
              _ForwardIterator1 __chunk_last,
              __decoupled_lookback_partition<size_t>* __nonexistent_lookback_partition) {
            // Reset the local storage by setting all flags to false
            __worker_ctx.__reset();

            // Populate the bitset for each of the elements in the chunk and get the occupancy
            __pstl::__count_and_cache_predicate_results(__chunk_first, __chunk_last, __pred, __worker_ctx);

            // Obtain the exclusive prefix
            __decoupled_lookback_partition< size_t >* __prev_partition = __nonexistent_lookback_partition - 1;
            size_t __exclusive_prefix =
                (__prev_partition->__acquire_available_status() & __decoupled_lookback_status_prefix_available)
                    ? __prev_partition->__inclusive_prefix()
                    : __pstl::__calculate_inclusive_prefix_of_partition(__prev_partition, plus<>{});

            // Copy the elements depending on the bitset flags
            _ForwardIterator2 __chunk_result =
                __pstl::__copy_if_flag_set(__chunk_first, __chunk_last, __result + __exclusive_prefix, __worker_ctx);

            // Report the position of the last iterator in the result range
            __out_iter = __chunk_result;
          };

      auto __ret = __cpu_traits<_Backend>::template __lookback_scan<size_t>(
          __first, __last, __prologue, __scan_head, __scan_middle, __scan_tail, __epilogue);
      if (!__ret)
        return nullopt;
      return __out_iter;
    } else {
      return std::copy_if(std::move(__first), std::move(__last), std::move(__result), std::move(__pred));
    }
  }
};

} // namespace __pstl
_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP_STD_VER >= 20

_LIBCPP_POP_MACROS

#endif // _LIBCPP___PSTL_CPU_ALGOS_COPY_IF_H
