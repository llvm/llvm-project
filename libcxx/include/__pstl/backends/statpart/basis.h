//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___PSTL_BACKENDS_STATPART_BASIS_H
#define _LIBCPP___PSTL_BACKENDS_STATPART_BASIS_H

#include <__algorithm/lower_bound.h>
#include <__algorithm/merge.h>
#include <__algorithm/min.h>
#include <__algorithm/move.h>
#include <__algorithm/upper_bound.h>
#include <__assert>
#include <__atomic/atomic.h>
#include <__config>
#include <__execution/is_execution_policy.h>
#include <__functional/identity.h>
#include <__functional/operations.h>
#include <__iterator/concepts.h>
#include <__iterator/distance.h>
#include <__iterator/iterator_traits.h>
#include <__iterator/move_iterator.h>
#include <__memory/allocator.h>
#include <__memory/construct_at.h>
#include <__memory/destroy.h>
#include <__memory/unique_ptr.h>
#include <__numeric/reduce.h>
#include <__optional/nullopt_t.h>
#include <__optional/optional.h>
#include <__thread/hardware_concurrency.h>
#include <__type_traits/is_assignable.h>
#include <__type_traits/is_constructible.h>
#include <__type_traits/is_trivially_copyable.h>
#include <__utility/empty.h>
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

// The backbone of the backend: applies the function to each of the indices [0, __chunk_count) in parallel.
// The actual number of parallel workers and the order of application is unspecified.
_LIBCPP_EXPORTED_FROM_ABI void
__apply(size_t __chunk_count, void* __context, void (*__func)(void* __context, size_t __chunk)) noexcept;

// The maximum number of partitions to be processed per CPU
constexpr inline size_t __max_partitions_per_cpu = 8;

// The minimum number of elements to sort serially in a single partition
constexpr inline size_t __min_sort_partition_size = 256;

constexpr inline size_t __lane_size = 64;

enum class __partitioning_status {
  // Returned when the partitioning result is not a nullopt.
  __ok,
  // Returned when there are not enough element to produce a single partition or the final number of partitions is fewer
  // than 2, i.e. there's no reason to process the input in parallel.
  __not_enough_elements,
  // There's fewer than 2 processors available in the system, or their number is unknown, which is treated as having
  // a single processor. In either case, fall back to serial processing.
  __no_parallelism,
  // Reserved for later: returned when memory could not be allocated for the stashed forward iterators.
  __bad_alloc
};

template <class _Iterator>
struct _LIBCPP_HIDE_FROM_ABI __partitioning_result;

template <class _Iterator>
struct _LIBCPP_HIDE_FROM_ABI __iterator_range {
  _Iterator __first;
  _Iterator __last;
};

// __iterator_partitions<Iter> holds a finite pre-determined set of partitions and allows querying each iterator range
// in O(1) time.
// TODO: Support forward iterators later via a second implicit template argument and its specializations.
template <class _Iterator>
struct _LIBCPP_HIDE_FROM_ABI __iterator_partitions {
  _LIBCPP_HIDE_FROM_ABI size_t __partitions_count() const { return __partitions_count_; }

  _LIBCPP_HIDE_FROM_ABI __iterator_range<_Iterator> __partition(size_t __n) const {
    _Iterator __first = __base_ + __n * __partition_size_ + (__n < __remainder_ ? __n : __remainder_);
    _Iterator __last  = __first + __partition_size_ + (__n < __remainder_ ? 1 : 0);
    return __iterator_range<_Iterator>{__first, __last};
  }

  _LIBCPP_HIDE_FROM_ABI _Iterator __end() const { return __base_ + __count_; }

private:
  _Iterator __base_;
  size_t __count_;
  size_t __partition_size_;
  size_t __remainder_;
  size_t __partitions_count_;

  template <class _Iter>
  friend __partitioning_result<_Iter> _LIBCPP_HIDE_FROM_ABI
  __partition(_Iter, _Iter, size_t, size_t, size_t, size_t) noexcept;
};

// This is basically a barebones std::expected<__iterator_partitions<_Iterator>, __partitioning_status>.
// __partition() uses this to avoid resorting to return arguments.
template <class _Iterator>
struct _LIBCPP_HIDE_FROM_ABI __partitioning_result {
  optional<__iterator_partitions<_Iterator>> __value_;
  __partitioning_status __status_ = __partitioning_status::__ok;
};

// This function allows basis.h to compile when _LIBCPP_HAS_THREADS=0
_LIBCPP_HIDE_FROM_ABI inline unsigned __thread_hardware_concurrency() noexcept {
#  if _LIBCPP_HAS_THREADS
  return std::__thread_hardware_concurrency();
#  else
  return 0;
#  endif // _LIBCPP_HAS_THREADS
}

// Splits the input iterator range into a finite set of partitions, such that every iterator sub-range can be queried in
// a O(1) time later.
// If partitioning fails, returns nullopt and reports the reason in __result.
// __min_partition_size specifies the minimal number of elements in every partition.
// __prefix_to_skip and __suffix_to_skip allows optionally trimming the input range by the specified number of elements
// before partitioning.
// __workers_count is normally implicitly queried from hardware_concurrency() and should be explicitly specified only in
// unit tests.
template <class _Iterator>
_LIBCPP_HIDE_FROM_ABI __partitioning_result<_Iterator>
__partition(_Iterator __first,
            _Iterator __last,
            size_t __min_partition_size = 1,
            size_t __prefix_to_skip     = 0,
            size_t __suffix_to_skip     = 0,
            size_t __workers_count      = __statpart::__thread_hardware_concurrency()) noexcept {
  // Forward iterators will be supported later.
  static_assert(__has_random_access_iterator_category_or_concept<_Iterator>::value);

  _LIBCPP_ASSERT_INTERNAL(__min_partition_size >= 1, "minimum partition size must be at least 1 element");

  if (__workers_count < 2) {
    // There's no available parallelism in the system, no reason to perform partitioning.
    return {nullopt, __partitioning_status::__no_parallelism};
  }

  size_t __count = static_cast<size_t>(std::distance(__first, __last));
  if (__count <= __prefix_to_skip + __suffix_to_skip) {
    return {nullopt, __partitioning_status::__not_enough_elements};
  }

  size_t __effective_count = __count - __prefix_to_skip - __suffix_to_skip;
  size_t __max_partitions  = __workers_count * __max_partitions_per_cpu;
  size_t __partition_count = std::min(__max_partitions, __effective_count / __min_partition_size);

  if (__partition_count < 2) {
    // not enough input data to process in parallel => no reason to build partitions
    return {nullopt, __partitioning_status::__not_enough_elements};
  }

  __iterator_partitions<_Iterator> __p;
  __p.__base_             = __first + __prefix_to_skip;
  __p.__count_            = __effective_count;
  __p.__partition_size_   = __effective_count / __partition_count;
  __p.__remainder_        = __effective_count % __partition_count;
  __p.__partitions_count_ = __partition_count;
  return {std::move(__p), __partitioning_status::__ok};
}

// for_each with explicit partitioning:
// calls __func in parallel with every iterator range provided in __partitions.
template <class _Iterator, class _Functor>
_LIBCPP_HIDE_FROM_ABI void __for_each_partition(const __iterator_partitions<_Iterator>& __partitions, _Functor __func) {
  struct __ctx_t {
    const __iterator_partitions<_Iterator>& __partitions_;
    _Functor& __func_;
  } __ctx{__partitions, __func};
  __statpart::__apply(__partitions.__partitions_count(), &__ctx, [](void* __context, size_t __partition) {
    __ctx_t& __ctx = *static_cast<__ctx_t*>(__context);
    auto __range   = __ctx.__partitions_.__partition(__partition);
    __ctx.__func_(__range.__first, __range.__last);
  });
}

// Calls __func(index) for every index in [0, __count) in parallel.
template <class _Functor>
_LIBCPP_HIDE_FROM_ABI void __for_each_index(size_t __count, _Functor __func) {
  __statpart::__apply(__count, &__func, [](void* __context, size_t __index) {
    (*static_cast<_Functor*>(__context))(__index);
  });
}

// for_each that calls __func in parallel for every partition index available in __partitions.
template <class _Iterator, class _Functor>
_LIBCPP_HIDE_FROM_ABI void
__for_each_partition_index(const __iterator_partitions<_Iterator>& __partitions, _Functor __func) {
  __statpart::__for_each_index(__partitions.__partitions_count(), std::move(__func));
}

// for_each with default implicit partitioning:
// calls __func in parallel with every element in [__first, __last), provided as iterator ranges.
template <class _Iterator, class _Functor>
_LIBCPP_HIDE_FROM_ABI optional<__empty> __for_each(_Iterator __first, _Iterator __last, _Functor __func) {
  if (__first == __last)
    return __empty{};
  auto __partitions = __statpart::__partition(__first, __last);
  if (__partitions.__value_) {
    __statpart::__for_each_partition(*__partitions.__value_, std::move(__func));
    return __empty{};
  } else if (__partitions.__status_ == __partitioning_status::__bad_alloc) {
    return nullopt;
  } else {
    __func(__first, __last);
    return __empty{};
  }
}

// Applies the 3-legged function to sub-ranges of the input ranges in parallel:
//   f(sub_range1_first, sub_range1_last, sub_range2_first)
template <class _RandomAccessIterator1, class _RandomAccessIterator2, class _BrickFunction>
_LIBCPP_HIDE_FROM_ABI optional<__empty> __for_each_iter_pair(
    _RandomAccessIterator1 __first1,
    _RandomAccessIterator1 __last1,
    _RandomAccessIterator2 __first2,
    _BrickFunction __f) {
  return __statpart::__for_each(
      __first1,
      __last1,
      [__first1, __first2, __f = std::move(__f)](
          _RandomAccessIterator1 __brick_first1, _RandomAccessIterator1 __brick_last1) {
        _RandomAccessIterator2 __brick_first2 = __first2 + (__brick_first1 - __first1);
        __f(__brick_first1, __brick_last1, __brick_first2);
      });
}

template <class _Index, class _Brick, class _Compare>
_LIBCPP_HIDE_FROM_ABI optional<_Index>
__find(_Index __first, _Index __last, _Brick __f, _Compare __comp, bool __b_first) {
  typedef typename std::iterator_traits<_Index>::difference_type _DifferenceType;
  const _DifferenceType __n      = __last - __first;
  _DifferenceType __initial_dist = __b_first ? __n : -1;
  std::atomic<_DifferenceType> __extremum(__initial_dist);
  // TODO: find out what is better here: parallel_for or parallel_reduce
  auto __res = __statpart::__for_each(__first, __last, [__comp, __f, __first, &__extremum](_Index __i, _Index __j) {
    // See "Reducing Contention Through Priority Updates", PPoPP '13, for discussion of
    // why using a shared variable scales fairly well in this situation.
    if (__comp(__i - __first, __extremum)) {
      _Index __result = __f(__i, __j);
      // If not '__last' returned then we found what we want so put this to extremum
      if (__result != __j) {
        const _DifferenceType __k = __result - __first;
        for (_DifferenceType __old = __extremum; __comp(__k, __old); __old = __extremum) {
          __extremum.compare_exchange_weak(__old, __k);
        }
      }
    }
  });
  if (!__res)
    return nullopt;
  return __extremum.load() != __initial_dist ? __first + __extremum.load() : __last;
}

template <class _RandomAccessIterator, class _Transform, class _Value, class _Combiner, class _Reduction>
_LIBCPP_HIDE_FROM_ABI optional<_Value> __transform_reduce(
    _RandomAccessIterator __first,
    _RandomAccessIterator __last,
    _Transform __transform,
    _Value __init,
    _Combiner __combiner,
    _Reduction __reduction) {
  if (__first == __last)
    return __init;

  // Split into partitions with at least 3 elements in each
  auto __partitions = __statpart::__partition(__first, __last, 3);
  if (!__partitions.__value_) {
    if (__partitions.__status_ == __partitioning_status::__bad_alloc) {
      // Failed to allocate memory for the partitions, propagate the error up.
      return nullopt;
    } else {
      // Not enough elements to run in parallel, switch to serial implementation.
      return __reduction(std::move(__first), std::move(__last), std::move(__init));
    }
  }

  auto __destroy = [__count = __partitions.__value_->__partitions_count()](_Value* __ptr) {
    std::destroy_n(__ptr, __count);
    std::allocator<_Value>().deallocate(__ptr, __count);
  };

  // TODO: use __uninitialized_buffer
  // TODO: allocate one element per worker instead of one element per chunk
  unique_ptr<_Value[], decltype(__destroy)> __values(
      std::allocator<_Value>().allocate(__partitions.__value_->__partitions_count()), __destroy);

  // __dispatch_apply is noexcept
  __statpart::__for_each_partition_index(*__partitions.__value_, [&](size_t __index) {
    auto __range = __partitions.__value_->__partition(__index);
    std::__construct_at(
        __values.get() + __index,
        __reduction(__range.__first + 2,
                    __range.__last,
                    __combiner(__transform(__range.__first), __transform(__range.__first + 1))));
  });

  return std::reduce(
      std::make_move_iterator(__values.get()),
      std::make_move_iterator(__values.get() + __partitions.__value_->__partitions_count()),
      std::move(__init),
      __combiner);
}

template <class _RandomAccessIterator1,
          class _RandomAccessIterator2,
          class _RandomAccessIterator3,
          class _Compare,
          class _LeafMerge>
_LIBCPP_HIDE_FROM_ABI optional<__empty>
__merge(_RandomAccessIterator1 __first1,
        _RandomAccessIterator1 __last1,
        _RandomAccessIterator2 __first2,
        _RandomAccessIterator2 __last2,
        _RandomAccessIterator3 __result,
        _Compare __comp,
        _LeafMerge __leaf_merge) noexcept {
  // Run the merge such that the longest range is partitioned.
  // __merge_sub_ranges swaps the ranges implicitly in case the second range was the longest one.
  auto __run =
      [__result](auto __long_first,
                 auto __long_last,
                 auto __short_first,
                 auto __short_last,
                 auto __find_cutpoint,
                 auto __merge_sub_ranges) -> optional<__empty> {
    // Partition the longest input range evenly.
    auto __partitions = __statpart::__partition(__long_first, __long_last);
    if (!__partitions.__value_) {
      if (__partitions.__status_ == __partitioning_status::__bad_alloc)
        return nullopt;
      // Not enough elements to run in parallel, merge everything serially.
      __merge_sub_ranges(__long_first, __long_last, __short_first, __short_last, __result);
      return __empty{};
    }

    // Process the partitions of the longest range in parallel.
    __statpart::__for_each_partition_index(*__partitions.__value_, [&](size_t __index) {
      // Get the sub-range inside the longer input range.
      auto __range = __partitions.__value_->__partition(__index);

      // Find the matching sub-range of the shorter input range: it will hold the elements that come after the last
      // element of the previous long partition and up to the last element of this partition.
      // This generally runs two binary searches in the smaller input range, edge partitions use existing bounds.
      auto __short_begin =
          __index == 0 ? __short_first : __find_cutpoint(__short_first, __short_last, __range.__first[-1]);
      auto __short_end =
          __index == __partitions.__value_->__partitions_count() - 1
              ? __short_last
              : __find_cutpoint(__short_first, __short_last, __range.__last[-1]);

      // Derive the position in the output range and merge the sub-ranges.
      auto __result_first = __result + (__range.__first - __long_first) + (__short_begin - __short_first);
      __merge_sub_ranges(__range.__first, __range.__last, __short_begin, __short_end, __result_first);
    });
    return __empty{};
  };

  if (__last1 - __first1 > __last2 - __first2) {
    // Preserve the order of the input ranges, partition the first range.
    // Equal elements of the second range go after the cut => use lower_bound when cutting.
    return __run(
        __first1,
        __last1,
        __first2,
        __last2,
        [&](auto __first, auto __last, const auto& __value) {
          return std::lower_bound(__first, __last, __value, __comp);
        },
        [&](auto __long_first, auto __long_last, auto __short_first, auto __short_last, auto __result_first) {
          __leaf_merge(__long_first, __long_last, __short_first, __short_last, __result_first, __comp);
        });
  } else {
    // Swap the order of the input ranges, partition the second range.
    // Equal elements of the first range go before the cut => use upper_bound when cutting.
    return __run(
        __first2,
        __last2,
        __first1,
        __last1,
        [&](auto __first, auto __last, const auto& __value) {
          return std::upper_bound(__first, __last, __value, __comp);
        },
        [&](auto __long_first, auto __long_last, auto __short_first, auto __short_last, auto __result_first) {
          __leaf_merge(__short_first, __short_last, __long_first, __long_last, __result_first, __comp);
        });
  }
}

template <class _RandomAccessIterator, class _Comp, class _LeafSort>
_LIBCPP_HIDE_FROM_ABI optional<__empty>
__stable_sort(_RandomAccessIterator __first, _RandomAccessIterator __last, _Comp __comp, _LeafSort __leaf_sort) {
  const auto __size = __last - __first;

  auto __partitions = __statpart::__partition(__first, __last, __min_sort_partition_size);
  if (!__partitions.__value_) {
    if (__partitions.__status_ == __partitioning_status::__bad_alloc)
      return nullopt;
    // Not enough elements to run in parallel, sort everything serially.
    __leaf_sort(__first, __last, __comp);
    return __empty{};
  }

  using _Value = __iterator_value_type<_RandomAccessIterator>;

  auto __destroy = [__size](_Value* __ptr) {
    std::destroy_n(__ptr, __size);
    std::allocator<_Value>().deallocate(__ptr, __size);
  };

  // TODO: use __uninitialized_buffer
  unique_ptr<_Value[], decltype(__destroy)> __values(std::allocator<_Value>().allocate(__size), __destroy);

  // Initialize all elements to a moved-from state
  // TODO: Don't do this - this can be done in the first merge - see https://llvm.org/PR63928
  std::__construct_at(__values.get(), std::move(*__first));
  for (__iterator_difference_type<_RandomAccessIterator> __i = 1; __i != __size; ++__i) {
    std::__construct_at(__values.get() + __i, std::move(__values.get()[__i - 1]));
  }
  *__first = std::move(__values.get()[__size - 1]);

  __statpart::__for_each_partition(
      *__partitions.__value_,
      [&__leaf_sort, &__comp](_RandomAccessIterator __chunk_first, _RandomAccessIterator __chunk_last) {
        __leaf_sort(std::move(__chunk_first), std::move(__chunk_last), __comp);
      });

  // A chunk span is a run of __chunk_span_size consecutive partitions, the last chunk can be shorter in case of odd
  // number of chunk spans. Each pass merges the chunk spans pairwise, so the chunk span size doubles and the chunk
  // count is halved (rounded up).
  const size_t __partition_count = __partitions.__value_->__partitions_count();
  size_t __chunk_span_size       = 1;
  size_t __chunk_spans           = __partition_count;

  // Offsets of the beginning and the end of the chunk from __first
  auto __chunk_begin = [&](size_t __chunk) {
    return __partitions.__value_->__partition(__chunk * __chunk_span_size).__first - __first;
  };
  auto __chunk_end = [&](size_t __chunk) {
    return __partitions.__value_->__partition(std::min((__chunk + 1) * __chunk_span_size, __partition_count) - 1)
               .__last -
           __first;
  };

  bool __objects_are_in_buffer = false;
  do {
    size_t __chunk_spans_pairs = (__chunk_spans + 1) / 2;
    auto __merge_chunks        = [&](auto __from_first, auto __to_first) {
      __statpart::__for_each_index(__chunk_spans_pairs, [&](size_t __span_pair_index) {
        size_t __left_chunk  = 2 * __span_pair_index;
        size_t __right_chunk = __left_chunk + 1;
        auto __pair_first    = __chunk_begin(__left_chunk);
        if (__right_chunk == __chunk_spans) {
          // The odd chunk has no counterpart to merge with => carry it over to the other buffer as-is.
          std::move(__from_first + __pair_first, __from_first + __chunk_end(__left_chunk), __to_first + __pair_first);
        } else {
          // Merge the two chunk spans into the other buffer
          auto __pair_mid  = __chunk_begin(__right_chunk);
          auto __pair_last = __chunk_end(__right_chunk);
          std::merge(std::make_move_iterator(__from_first + __pair_first),
                     std::make_move_iterator(__from_first + __pair_mid),
                     std::make_move_iterator(__from_first + __pair_mid),
                     std::make_move_iterator(__from_first + __pair_last),
                     __to_first + __pair_first,
                     __comp);
        }
      });
    };

    if (__objects_are_in_buffer)
      __merge_chunks(__values.get(), __first);
    else
      __merge_chunks(__first, __values.get());
    __objects_are_in_buffer = !__objects_are_in_buffer;

    __chunk_span_size *= 2;
    __chunk_spans = __chunk_spans_pairs;
  } while (__chunk_spans > 1);

  if (__objects_are_in_buffer) {
    std::move(__values.get(), __values.get() + __size, __first);
  }

  return __empty{};
}

} // namespace __statpart

} // namespace __pstl
_LIBCPP_END_EXPLICIT_ABI_ANNOTATIONS
_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP_STD_VER >= 17

_LIBCPP_POP_MACROS

#endif // _LIBCPP___PSTL_BACKENDS_STATPART_BASIS_H
