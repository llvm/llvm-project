//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___PSTL_STATPART_H
#define _LIBCPP___PSTL_STATPART_H

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
namespace __statpart {

// The backbone for the backend: applies the function to each of the indices [0, __chunk_count] in parallel.
// The actual number of parallel workers and the order of application is unspecified.
_LIBCPP_EXPORTED_FROM_ABI void
__apply(size_t __chunk_count, void* __context, void (*__func)(void* __context, size_t __chunk)) noexcept;

// The maximum number of partitions to be processed per each CPU
constexpr inline size_t __max_partitions_per_cpu = 8;

// The minimum number of elements to sort serially in a single partition
constexpr inline size_t __min_sort_partition_size = 256;

enum class __partitioning_result {
  // returned when the partitioning result is not nullopt
  __ok,
  // returned when there is not enough element to produce a single partition or the final number of partitions is less
  // than 2, i.e. there's no reason to process in parallel
  __not_enough_elements,
  // later: returned memory couldn't be allocated for the stashed forward iterators
  __bad_alloc
};

template <class _Iterator>
struct _LIBCPP_HIDE_FROM_ABI __iterator_range {
  _Iterator __first;
  _Iterator __last;
};

// Later to support forward iterators via a second implicit template argument.
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
  friend optional<__iterator_partitions<_Iter>>
  __partition(_Iter, _Iter, __partitioning_result&, size_t, size_t, size_t) noexcept;
};

template <class _ForwardIterator>
_LIBCPP_HIDE_FROM_ABI optional<__iterator_partitions<_ForwardIterator>>
__partition(_ForwardIterator __first,
            _ForwardIterator __last,
            __partitioning_result& __result,
            size_t __min_partition_size = 1,
            size_t __prefix_to_skip     = 0,
            size_t __suffix_to_skip     = 0) noexcept {
  // forward iteratos to be supported later
  static_assert(__has_random_access_iterator_category_or_concept<_ForwardIterator>::value);
  // TODO: assert( __min_partition_size > 0 )?

  size_t __count = static_cast<size_t>(std::distance(__first, __last));
  if (__count < __prefix_to_skip + __suffix_to_skip) {
    __result = __partitioning_result::__not_enough_elements;
    return {};
  }

  size_t __effective_count = __count - __prefix_to_skip - __suffix_to_skip;
  size_t __workers_count   = thread::hardware_concurrency();
  size_t __max_partitions  = __workers_count * __max_partitions_per_cpu;
  size_t __partition_count = std::min(__max_partitions, __effective_count / __min_partition_size);

  if (__partition_count < 2) {
    // TODO: verify that's what we want
    __result = __partitioning_result::__not_enough_elements;
    return {}; // not enough input data to process in parallel => no reason to build partitions
  }

  __iterator_partitions<_ForwardIterator> __p;
  __p.__base_             = __first + __prefix_to_skip;
  __p.__count_            = __count;
  __p.__partition_size_   = __count / __partition_count;
  __p.__remainder_        = __count % __partition_count;
  __p.__partitions_count_ = __partition_count;
  return __p;
}

// for_each with explicit partitioning
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

// Calls __func(index) for every index in [0, __count) in parallel
template <class _Functor>
_LIBCPP_HIDE_FROM_ABI void __for_each_index(size_t __count, _Functor __func) {
  __statpart::__apply(__count, &__func, [](void* __context, size_t __index) {
    (*static_cast<_Functor*>(__context))(__index);
  });
}

// TODO: ...
template <class _Iterator, class _Functor>
_LIBCPP_HIDE_FROM_ABI void
__for_each_partition_index(const __iterator_partitions<_Iterator>& __partitions, _Functor __func) {
  __statpart::__for_each_index(__partitions.__partitions_count(), std::move(__func));
}

// for_each with default implicit partitioning
template <class _Iterator, class _Functor>
_LIBCPP_HIDE_FROM_ABI optional<__empty> __for_each(_Iterator __first, _Iterator __last, _Functor __func) {
  if (__first == __last)
    return __empty{};
  __partitioning_result __partition_result;
  auto __partitions = __statpart::__partition(__first, __last, __partition_result);
  if (__partitions) {
    __statpart::__for_each_partition(*__partitions, std::move(__func));
    return __empty{};
  } else if (__partition_result == __partitioning_result::__bad_alloc) {
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
  return __for_each(
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
  __partitioning_result __partitioning_result;
  auto __partitions = __statpart::__partition(__first, __last, __partitioning_result, 3);
  if (!__partitions) {
    if (__partitioning_result == __partitioning_result::__bad_alloc) {
      // Failed to allocate memory for the partitions, propagate the error up.
      return nullopt;
    } else {
      // Not enough elements to run in parallel, switch to serial implementation.
      return __reduction(std::move(__first), std::move(__last), std::move(__init));
    }
  }

  auto __destroy = [__count = __partitions->__partitions_count()](_Value* __ptr) {
    std::destroy_n(__ptr, __count);
    std::allocator<_Value>().deallocate(__ptr, __count);
  };

  // TODO: use __uninitialized_buffer
  // TODO: allocate one element per worker instead of one element per chunk
  unique_ptr<_Value[], decltype(__destroy)> __values(
      std::allocator<_Value>().allocate(__partitions->__partitions_count()), __destroy);

  // __dispatch_apply is noexcept
  __statpart::__for_each_partition_index(*__partitions, [&](size_t __index) {
    auto __range = __partitions->__partition(__index);
    std::__construct_at(
        __values.get() + __index,
        __reduction(__range.__first + 2,
                    __range.__last,
                    __combiner(__transform(__range.__first), __transform(__range.__first + 1))));
  });

  return std::reduce(
      std::make_move_iterator(__values.get()),
      std::make_move_iterator(__values.get() + __partitions->__partitions_count()),
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
    __partitioning_result __partitioning_result;
    auto __partitions = __statpart::__partition(__long_first, __long_last, __partitioning_result);
    if (!__partitions) {
      if (__partitioning_result == __partitioning_result::__bad_alloc)
        return nullopt;
      // Not enough elements to run in parallel, merge everything serially.
      __merge_sub_ranges(__long_first, __long_last, __short_first, __short_last, __result);
      return __empty{};
    }

    // Process the partitions of the longest range in parallel.
    __statpart::__for_each_partition_index(*__partitions, [&](size_t __index) {
      // Get the sub-range inside the longer input range.
      auto __range = __partitions->__partition(__index);

      // Find the matching sub-range of the shorter input range: it will hold the elements that come after the last
      // element of the previous long partition and up to the last element of this partition.
      // This generally runs two binary searches in the smaller input range, edge partitions use existing bounds.
      auto __short_begin =
          __index == 0 ? __short_first : __find_cutpoint(__short_first, __short_last, __range.__first[-1]);
      auto __short_end =
          __index == __partitions->__partitions_count() - 1
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

  __partitioning_result __partitioning_result;
  auto __partitions = __statpart::__partition(__first, __last, __partitioning_result, __min_sort_partition_size);
  if (!__partitions) {
    if (__partitioning_result == __partitioning_result::__bad_alloc)
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
      *__partitions, [&__leaf_sort, &__comp](_RandomAccessIterator __chunk_first, _RandomAccessIterator __chunk_last) {
        __leaf_sort(std::move(__chunk_first), std::move(__chunk_last), __comp);
      });

  // A chunk span is a run of __chunk_span_size consecutive partitions, the last chunk can be shorter in case of odd
  // number of chunk spans. Each pass merges the chunk spans pairwise, so the chunk span size doubles and the chunk
  // count is halved (rounded up).
  const size_t __partition_count = __partitions->__partitions_count();
  size_t __chunk_span_size       = 1;
  size_t __chunk_spans           = __partition_count;

  // Offsets of the beginning and the end of the chunk from __first
  auto __chunk_begin = [&](size_t __chunk) {
    return __partitions->__partition(__chunk * __chunk_span_size).__first - __first;
  };
  auto __chunk_end = [&](size_t __chunk) {
    return __partitions->__partition(std::min((__chunk + 1) * __chunk_span_size, __partition_count) - 1).__last -
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

template <class _ExecutionPolicy>
struct __find_end<__statpart_backend_tag, _ExecutionPolicy> {
  template <class _Policy, class _ForwardIterator1, class _ForwardIterator2, class _BinaryPredicate>
  _LIBCPP_HIDE_FROM_ABI optional<_ForwardIterator1>
  operator()(_Policy&&,
             _ForwardIterator1 __first1,
             _ForwardIterator1 __last1,
             _ForwardIterator2 __first2,
             _ForwardIterator2 __last2,
             _BinaryPredicate __pred) const noexcept {
    if constexpr (__is_parallel_execution_policy_v<_ExecutionPolicy> &&
                  __has_random_access_iterator_category_or_concept<_ForwardIterator1>::value &&
                  __has_random_access_iterator_category_or_concept<_ForwardIterator2>::value) {
      typedef typename std::iterator_traits<_ForwardIterator1>::difference_type _DifferenceType;
      _DifferenceType __size2 = __last2 - __first2; // The length of the needle to search for.
      if (__size2 == 0) {
        return __last1; // If the needle length is zero, the last iterator is returned.
      }
      _DifferenceType __size1 = __last1 - __first1;
      if (__size1 < __size2) {
        return __last1; // The range is too small to contain the requested number of consecutive elements.
      }
      // Calculate the length of the tail where a potential match cannot start by definition.
      _DifferenceType __crop = __size2 - 1;
      // We're only interested in the range where a potential match can start: [first, last - crop)
      _ForwardIterator1 __last1_cropped = __last1 - __crop;
      // Run a parallel chunked find_if, covering the range where a potential match can start.
      auto __res = __statpart::__find(
          __first1,
          __last1_cropped,
          [__first2, __last2, __crop, &__pred](_ForwardIterator1 __brick_first, _ForwardIterator1 __brick_last) {
            // Uncrop the range to allow std::find_end to find a full match, which can go beyond __brick_last.
            _ForwardIterator1 __brick_last_uncropped = __brick_last + __crop;
            // Run a serial std::find_end inside each of the chunks in parallel.
            _ForwardIterator1 __ret = std::find_end(__brick_first, __brick_last_uncropped, __first2, __last2, __pred);
            // The returned iterator is either a match inside [__brick_first, __brick_last) or a miss encoded as
            // __brick_last_uncropped. Return the miss as __brick_last to conform to expectations of __parallel_find().
            return __ret == __brick_last_uncropped ? __brick_last : __ret;
          },
          greater<>{}, // `greater` here means the highest index among the matches
          false        // `false` here means we want the last match, not the first
      );
      if (!__res) {
        return std::nullopt; // Failed to run the algorithm, propagate the error.
      }
      if (*__res == __last1_cropped) {
        return __last1; // No match was found in the range.
      }
      return *__res; // Return the successful match.
    } else {
      // Non-random access iterators cannot be processed in parallel, fall back to the sequential implementation.
      return std::find_end(
          std::move(__first1), std::move(__last1), std::move(__first2), std::move(__last2), std::move(__pred));
    }
  }
};

template <class _ExecutionPolicy>
struct __for_each<__statpart_backend_tag, _ExecutionPolicy> {
  template <class _Policy, class _ForwardIterator, class _Function>
  _LIBCPP_HIDE_FROM_ABI optional<__empty>
  operator()(_Policy&&, _ForwardIterator __first, _ForwardIterator __last, _Function __func) const noexcept {
    if constexpr (__is_parallel_execution_policy_v<_ExecutionPolicy> &&
                  __has_random_access_iterator_category_or_concept<_ForwardIterator>::value) {
      return __statpart::__for_each(
          std::move(__first), std::move(__last), [&](auto __partition_first, auto __partition_last) {
            std::for_each(__partition_first, __partition_last, __func);
          });
    } else {
      std::for_each(std::move(__first), std::move(__last), std::move(__func));
      return __empty{};
    }
  }
};

template <class _ExecutionPolicy>
struct __find_if<__statpart_backend_tag, _ExecutionPolicy> {
  template <class _Policy, class _ForwardIterator, class _Predicate>
  _LIBCPP_HIDE_FROM_ABI optional<_ForwardIterator>
  operator()(_Policy&&, _ForwardIterator __first, _ForwardIterator __last, _Predicate __pred) const noexcept {
    if constexpr (__is_parallel_execution_policy_v<_ExecutionPolicy> &&
                  __has_random_access_iterator_category_or_concept<_ForwardIterator>::value) {
      return __statpart::__find(
          std::move(__first),
          std::move(__last),
          [&__pred](_ForwardIterator __brick_first, _ForwardIterator __brick_last) {
            return std::find_if(__brick_first, __brick_last, __pred);
          },
          less<>{},
          true);
    } else {
      return std::find_if(std::move(__first), std::move(__last), std::move(__pred));
    }
  }
};

template <class _ExecutionPolicy>
struct __is_heap_until<__statpart_backend_tag, _ExecutionPolicy> {
  template <class _Policy, class _RandomAccessIterator, class _Comp>
  _LIBCPP_HIDE_FROM_ABI optional<_RandomAccessIterator>
  operator()(_Policy&&, _RandomAccessIterator __first, _RandomAccessIterator __last, _Comp __comp) const noexcept {
    if constexpr (__is_parallel_execution_policy_v<_ExecutionPolicy>) {
      if (__last - __first < 2)
        return __last; // Any sequence with less than 2 elements is a heap
      // Run a parallel find in chunks over [first+1, last) to validate every element except the root.
      return __statpart::__find(
          __first + 1,
          __last,
          [__first, &__comp](_RandomAccessIterator __child_first, _RandomAccessIterator __child_last) {
            // This algorithm assumes that __find() will never pass an empty brick,
            // i.e. that __child_first != __child_last.
            using _DifferenceType = typename std::iterator_traits<_RandomAccessIterator>::difference_type;

            // Derive the indices of the children and the iterators of their parents
            _DifferenceType __child_first_idx   = __child_first - __first;
            _DifferenceType __child_last_idx    = __child_last - __first;
            _RandomAccessIterator __parent      = __first + (__child_first_idx - 1) / 2;
            _RandomAccessIterator __parent_last = __first + (__child_last_idx - 1) / 2;

            // If we're starting from a right child => process this element separately in a prologue
            if (__child_first_idx % 2 == 0) {
              if (__comp(*__parent, *__child_first)) // Check the right child
                return __child_first;
              ++__parent;
              ++__child_first;
            }

            // Iterate over the parents and check their left and right children
            for (; __parent != __parent_last; ++__parent) {
              if (__comp(*__parent, *__child_first)) // Check the left child
                return __child_first;
              ++__child_first;

              if (__comp(*__parent, *__child_first)) // Check the right child
                return __child_first;
              ++__child_first;
            }

            // If we're ending with a right child => __parent_last also includes the left child.
            // Process this accessible element separately in an epilogue.
            if (__child_last_idx % 2 == 0) {
              if (__comp(*__parent, *__child_first)) // Check the left child
                return __child_first;
            }

            return __child_last; // No violations found in this brick
          },
          less<>{}, // `less` here means the lowest index among the matches
          true      // `true` here means we want the first match, not the last
      );
    } else {
      return std::is_heap_until(std::move(__first), std::move(__last), std::move(__comp));
    }
  }
};

template <class _ExecutionPolicy>
struct __merge<__statpart_backend_tag, _ExecutionPolicy> {
  template <class _Policy, class _ForwardIterator1, class _ForwardIterator2, class _ForwardOutIterator, class _Comp>
  _LIBCPP_HIDE_FROM_ABI optional<_ForwardOutIterator> operator()(
      _Policy&&,
      _ForwardIterator1 __first1,
      _ForwardIterator1 __last1,
      _ForwardIterator2 __first2,
      _ForwardIterator2 __last2,
      _ForwardOutIterator __result,
      _Comp __comp) const noexcept {
    if constexpr (__is_parallel_execution_policy_v<_ExecutionPolicy> &&
                  __has_random_access_iterator_category_or_concept<_ForwardIterator1>::value &&
                  __has_random_access_iterator_category_or_concept<_ForwardIterator2>::value &&
                  __has_random_access_iterator_category_or_concept<_ForwardOutIterator>::value) {
      auto __res = __statpart::__merge(
          __first1,
          __last1,
          __first2,
          __last2,
          __result,
          __comp,
          [](_ForwardIterator1 __brick_first1,
             _ForwardIterator1 __brick_last1,
             _ForwardIterator2 __brick_first2,
             _ForwardIterator2 __brick_last2,
             _ForwardOutIterator __brick_result,
             _Comp __g_comp) {
            std::merge(std::move(__brick_first1),
                       std::move(__brick_last1),
                       std::move(__brick_first2),
                       std::move(__brick_last2),
                       std::move(__brick_result),
                       std::move(__g_comp));
          });
      if (!__res)
        return nullopt;
      return __result + (__last1 - __first1) + (__last2 - __first2);
    } else {
      return std::merge(__first1, __last1, __first2, __last2, __result, __comp);
    }
  }
};

template <class _ExecutionPolicy>
struct __min_element<__statpart_backend_tag, _ExecutionPolicy> {
  template <class _Policy, class _ForwardIterator, class _Compare>
  _LIBCPP_HIDE_FROM_ABI optional<_ForwardIterator>
  operator()(_Policy&&, _ForwardIterator __first, _ForwardIterator __last, _Compare __comp) const noexcept {
    if constexpr (__is_parallel_execution_policy_v<_ExecutionPolicy> &&
                  __has_random_access_iterator_category_or_concept<_ForwardIterator>::value) {
      if (__first == __last) {
        return __last; // nothing to do
      }

      _ForwardIterator __init = __first;
      ++__first;
      if (__first == __last) {
        return __init; // the only element is the minimum
      }

      // A reduction that returns an iterator pointing to the lowest element, left bias in case of a tie
      auto __iter_reduce = [&__comp](_ForwardIterator __lhs, _ForwardIterator __rhs) {
        return __comp(*__rhs, *__lhs) ? __rhs : __lhs;
      };

      // Perform a parallel reduction of iterators [first+1, last) with 'first' as init.
      return __statpart::__transform_reduce(
          std::move(__first),
          std::move(__last),
          __identity{},      // No transformations
          std::move(__init), // Use the first iterator as the init element
          __iter_reduce,     // Reduction of 2 elements
          [&__iter_reduce, &__comp](auto __brick_first, auto __brick_last, auto __brick_init) {
            // Reduction of an iterator range + init element: use the serial version to find the minimum among
            // the iterators and then reduce with the init element.
            return __iter_reduce(__brick_init, std::min_element(__brick_first, __brick_last, __comp));
          });
    } else {
      // Non-random access iterators cannot be processed in parallel, fall back to the sequential implementation.
      return std::min_element(std::move(__first), std::move(__last), std::move(__comp));
    }
  }
};

template <class _ExecutionPolicy>
struct __minmax_element<__statpart_backend_tag, _ExecutionPolicy> {
  template <class _Policy, class _ForwardIterator, class _Compare>
  _LIBCPP_HIDE_FROM_ABI optional<pair<_ForwardIterator, _ForwardIterator>>
  operator()(_Policy&&, _ForwardIterator __first, _ForwardIterator __last, _Compare __comp) const noexcept {
    if constexpr (__is_parallel_execution_policy_v<_ExecutionPolicy> &&
                  __has_random_access_iterator_category_or_concept<_ForwardIterator>::value) {
      using _IterPair = pair<_ForwardIterator, _ForwardIterator>;

      if (__first == __last) {
        return _IterPair{__last, __last}; // Nothing to do
      }

      _IterPair __init = {__first, __first};
      ++__first;
      if (__first == __last) {
        return __init; // The only element is both the minimum and the maximum
      }

      // A reduction that returns a pair of iterators pointing to the minimum and maximum elements.
      // In a case of a tie the minimum iterators are biased left and the maximum iterators are biased right.
      auto __iter_reduce = [&__comp](_IterPair __lhs, _IterPair __rhs) {
        return _IterPair{__comp(*__rhs.first, *__lhs.first) ? __rhs.first : __lhs.first,
                         __comp(*__rhs.second, *__lhs.second) ? __lhs.second : __rhs.second};
      };

      // Perform a parallel reduction of iterators [first+1, last) with {first, first} as init.
      return __statpart::__transform_reduce(
          std::move(__first),
          std::move(__last),
          [](auto __it) { return _IterPair{__it, __it}; }, // Transform an iterator into an iterator pair
          std::move(__init),                               // Use the pair of first iterators as the init element
          __iter_reduce,                                   // Reduction of 2 minmax pairs
          [&__iter_reduce, &__comp](auto __brick_first, auto __brick_last, auto __brick_init) {
            // Reduction of an iterator range + init element: use the serial version to find the minmax among
            // the iterators and then reduce it with the init element.
            return __iter_reduce(__brick_init, std::minmax_element(__brick_first, __brick_last, __comp));
          });
    } else {
      // Non-random access iterators cannot be processed in parallel, fall back to the sequential implementation.
      return std::minmax_element(std::move(__first), std::move(__last), std::move(__comp));
    }
  }
};

template <class _ExecutionPolicy>
struct __mismatch<__statpart_backend_tag, _ExecutionPolicy> {
  template <class _Policy, class _ForwardIterator1, class _ForwardIterator2, class _Predicate>
  _LIBCPP_HIDE_FROM_ABI optional<pair<_ForwardIterator1, _ForwardIterator2>>
  operator()(_Policy&&,
             _ForwardIterator1 __first1,
             _ForwardIterator1 __last1,
             _ForwardIterator2 __first2,
             _ForwardIterator2 __last2,
             _Predicate __pred) const noexcept {
    if constexpr (__is_parallel_execution_policy_v<_ExecutionPolicy> &&
                  __has_random_access_iterator_category_or_concept<_ForwardIterator1>::value &&
                  __has_random_access_iterator_category_or_concept<_ForwardIterator2>::value) {
      // Look for a mismatch only in the prefix of the two ranges.
      auto __n = std::min(__last1 - __first1, __last2 - __first2);
      // Find a position in the first range where the predicate is false against the corresponding position in the
      // second range.
      auto __res = __statpart::__find(
          __first1,
          __first1 + __n,
          [&__pred, __first1, __first2](_ForwardIterator1 __brick_first1, _ForwardIterator1 __brick_last1) {
            // Run the sequential mismatch algorithm on these ranges:
            //   [__brick_first1, __brick_last1) and
            //   [__first2 + (__brick_first1 - __first1), __first2 + (__brick_last1 - __first1))
            auto __brick_first2 = __first2 + (__brick_first1 - __first1);
            return std::mismatch(std::move(__brick_first1), std::move(__brick_last1), std::move(__brick_first2), __pred)
                .first;
          },
          less<>{}, // `less` here means the lowest index among the mismatches
          true      // `true` here means we want the first mismatch, not the last
      );
      if (!__res) {
        return std::nullopt; // Failed to run the algorithm, propagate the error.
      }
      auto __idx = *__res - __first1;
      return pair<_ForwardIterator1, _ForwardIterator2>{std::move(*__res), __first2 + __idx};
    } else {
      // Non-random access iterators cannot be processed in parallel, fall back to the sequential implementation.
      // Unsequenced execution is also implicitly covered by the sequential implementation.
      return std::mismatch(
          std::move(__first1), std::move(__last1), std::move(__first2), std::move(__last2), std::move(__pred));
    }
  }
};

template <class _ExecutionPolicy>
struct __reverse<__statpart_backend_tag, _ExecutionPolicy> {
  template <class _Policy, class _BidirectionalIterator>
  _LIBCPP_HIDE_FROM_ABI optional<__empty>
  operator()(_Policy&&, _BidirectionalIterator __first, _BidirectionalIterator __last) const noexcept {
    if constexpr (__is_parallel_execution_policy_v<_ExecutionPolicy> &&
                  __has_random_access_iterator_category_or_concept<_BidirectionalIterator>::value) {
      // Perform a chunked for_each on the first half of the range.
      // Odd-sized ranges will leave the middle element untouched, which is correct for this algorithm:
      // its reversed position is the same.
      auto __n = (__last - __first) / 2;
      return __statpart::__for_each(
          __first, __first + __n, [__first, __last](_BidirectionalIterator __i, _BidirectionalIterator __j) {
            // Derive the last position of the mirrored range by counting from the end.
            _BidirectionalIterator __mirror_last = __last - (__i - __first);
            // Swap the elements in the range of the first half with their mirrored counterparts in the second half.
            std::swap_ranges(std::move(__i),
                             std::move(__j),
                             std::reverse_iterator<_BidirectionalIterator>(std::move(__mirror_last)));
          });
    } else {
      // Non-random access iterators currently cannot be processed in parallel, use the sequential implementation.
      std::reverse(std::move(__first), std::move(__last));
      return __empty{};
    }
  }
};

template <class _ExecutionPolicy>
struct __search<__statpart_backend_tag, _ExecutionPolicy> {
  template <class _Policy, class _ForwardIterator1, class _ForwardIterator2, class _BinaryPredicate>
  _LIBCPP_HIDE_FROM_ABI optional<_ForwardIterator1>
  operator()(_Policy&&,
             _ForwardIterator1 __first1,
             _ForwardIterator1 __last1,
             _ForwardIterator2 __first2,
             _ForwardIterator2 __last2,
             _BinaryPredicate __pred) const noexcept {
    if constexpr (__is_parallel_execution_policy_v<_ExecutionPolicy> &&
                  __has_random_access_iterator_category_or_concept<_ForwardIterator1>::value &&
                  __has_random_access_iterator_category_or_concept<_ForwardIterator2>::value) {
      typedef typename std::iterator_traits<_ForwardIterator1>::difference_type _DifferenceType;
      _DifferenceType __size2 = __last2 - __first2; // The length of the needle to search for.
      if (__size2 == 0) {
        return __first1; // If the needle length is zero, the first iterator is returned.
      }
      _DifferenceType __size1 = __last1 - __first1;
      if (__size1 < __size2) {
        return __last1; // The range is too small to contain the requested number of consecutive elements.
      }
      // Calculate the length of the tail where a potential match cannot start by definition.
      _DifferenceType __crop = __size2 - 1;
      // We're only interested in the range where a potential match can start: [first, last - crop)
      _ForwardIterator1 __last1_cropped = __last1 - __crop;
      // Run a parallel chunked find_if, covering the range where a potential match can start.
      auto __res = __statpart::__find(
          __first1,
          __last1_cropped,
          [__first2, __last2, __crop, &__pred](_ForwardIterator1 __brick_first, _ForwardIterator1 __brick_last) {
            // Uncrop the range to allow std::search to find a full match, which can go beyond __brick_last.
            _ForwardIterator1 __brick_last_uncropped = __brick_last + __crop;
            // Run a serial std::search inside each of the chunks in parallel.
            _ForwardIterator1 __ret = std::search(__brick_first, __brick_last_uncropped, __first2, __last2, __pred);
            // The returned iterator is either a match inside [__brick_first, __brick_last) or a miss encoded as
            // __brick_last_uncropped. Return the miss as __brick_last to conform to expectations of __find().
            return __ret == __brick_last_uncropped ? __brick_last : __ret;
          },
          less<>{}, // `less` here means the lowest index among the matches
          true      // `true` here means we want the first match, not the last
      );
      if (!__res) {
        return std::nullopt; // Failed to run the algorithm, propagate the error.
      }
      if (*__res == __last1_cropped) {
        return __last1; // No match was found in the range.
      }
      return *__res; // Return the successful match.
    } else {
      // Non-random access iterators cannot be processed in parallel, fall back to the sequential implementation.
      return std::search(
          std::move(__first1), std::move(__last1), std::move(__first2), std::move(__last2), std::move(__pred));
    }
  }
};

template <class _ExecutionPolicy>
struct __search_n<__statpart_backend_tag, _ExecutionPolicy> {
  template <class _Policy, class _ForwardIterator, class _Size, class _Tp, class _Predicate>
  _LIBCPP_HIDE_FROM_ABI optional<_ForwardIterator>
  operator()(_Policy&&,
             _ForwardIterator __first,
             _ForwardIterator __last,
             _Size __count,
             const _Tp& __value,
             _Predicate __pred) const noexcept {
    if constexpr (__is_parallel_execution_policy_v<_ExecutionPolicy> &&
                  __has_random_access_iterator_category_or_concept<_ForwardIterator>::value) {
      typedef typename std::iterator_traits<_ForwardIterator>::difference_type _DifferenceType;
      _DifferenceType __integral_count = std::__convert_to_integral(__count);
      if (__integral_count <= 0) {
        return __first; // If the count is non-positive, the first iterator is returned.
      }
      _DifferenceType __size = __last - __first;
      if (__size < __integral_count) {
        return __last; // The range is too small to contain the requested number of consecutive elements.
      }
      // Calculate the length of the tail where a potential match cannot start by definition.
      _DifferenceType __crop = __integral_count - 1;
      // We're only interested in the range where a potential match can start: [first, last - crop)
      _ForwardIterator __last2 = __last - __crop;
      // Run a parallel chunked find_if, covering the range where a potential match can start.
      auto __res = __statpart::__find(
          __first,
          __last2,
          [__integral_count, __crop, &__value, &__pred](_ForwardIterator __brick_first, _ForwardIterator __brick_last) {
            // Uncrop the range to allow std::search_n to find a full match, which can go beyond __brick_last.
            _ForwardIterator __brick_last_uncropped = __brick_last + __crop;
            // Run a serial std::search_n inside each of the chunks in parallel.
            _ForwardIterator __ret =
                std::search_n(__brick_first, __brick_last_uncropped, __integral_count, __value, __pred);
            // The returned iterator is either a match inside [__brick_first, __brick_last) or a miss encoded as
            // __brick_last_uncropped. Return the miss as __brick_last to conform to expectations of __find().
            return __ret == __brick_last_uncropped ? __brick_last : __ret;
          },
          less<>{}, // `less` here means the lowest index among the matches
          true      // `true` here means we want the first match, not the last
      );
      if (!__res) {
        return std::nullopt; // Failed to run the algorithm, propagate the error.
      }
      if (*__res == __last2) {
        return __last; // No match was found in the range.
      }
      return *__res; // Return the successful match.
    } else {
      // Non-random access iterators cannot be processed in parallel, fall back to the sequential implementation.
      return std::search_n(std::move(__first), std::move(__last), __count, __value, std::move(__pred));
    }
  }
};

template <class _ExecutionPolicy>
struct __stable_sort<__statpart_backend_tag, _ExecutionPolicy> {
  template <class _Policy, class _RandomAccessIterator, class _Comp>
  _LIBCPP_HIDE_FROM_ABI optional<__empty>
  operator()(_Policy&&, _RandomAccessIterator __first, _RandomAccessIterator __last, _Comp __comp) const noexcept {
    if constexpr (__is_parallel_execution_policy_v<_ExecutionPolicy>) {
      return __statpart::__stable_sort(
          __first, __last, __comp, [](_RandomAccessIterator __g_first, _RandomAccessIterator __g_last, _Comp __g_comp) {
            std::stable_sort(__g_first, __g_last, __g_comp);
          });
    } else {
      std::stable_sort(__first, __last, __comp);
      return __empty{};
    }
  }
};

template <class _ExecutionPolicy>
struct __swap_ranges<__statpart_backend_tag, _ExecutionPolicy> {
  template <class _Policy, class _ForwardIterator1, class _ForwardIterator2>
  _LIBCPP_HIDE_FROM_ABI optional<_ForwardIterator2> operator()(
      _Policy&&, _ForwardIterator1 __first1, _ForwardIterator1 __last1, _ForwardIterator2 __first2) const noexcept {
    if constexpr (__is_parallel_execution_policy_v<_ExecutionPolicy> &&
                  __has_random_access_iterator_category_or_concept<_ForwardIterator1>::value &&
                  __has_random_access_iterator_category_or_concept<_ForwardIterator2>::value) {
      auto __res = __statpart::__for_each_iter_pair(
          __first1,
          __last1,
          __first2,
          [](_ForwardIterator1 __brick_first1, _ForwardIterator1 __brick_last1, _ForwardIterator2 __brick_first2) {
            std::swap_ranges(std::move(__brick_first1), std::move(__brick_last1), std::move(__brick_first2));
          });
      if (!__res) {
        return std::nullopt; // Failed to run the algorithm, propagate the error.
      }
      return __first2 + (__last1 - __first1);
    } else {
      // Non-random access iterators cannot be processed in parallel, fall back to the sequential implementation.
      return std::swap_ranges(std::move(__first1), std::move(__last1), std::move(__first2));
    }
  }
};

template <class _ExecutionPolicy>
struct __transform<__statpart_backend_tag, _ExecutionPolicy> {
  template <class _Policy, class _ForwardIterator, class _ForwardOutIterator, class _UnaryOperation>
  _LIBCPP_HIDE_FROM_ABI optional<_ForwardOutIterator> operator()(
      _Policy&&, _ForwardIterator __first, _ForwardIterator __last, _ForwardOutIterator __result, _UnaryOperation __op)
      const noexcept {
    if constexpr (__is_parallel_execution_policy_v<_ExecutionPolicy> &&
                  __has_random_access_iterator_category_or_concept<_ForwardIterator>::value &&
                  __has_random_access_iterator_category_or_concept<_ForwardOutIterator>::value) {
      __statpart::__for_each(
          __first, __last, [__op, __first, __result](_ForwardIterator __brick_first, _ForwardIterator __brick_last) {
            return std::transform(__brick_first, __brick_last, __result + (__brick_first - __first), __op);
          });
      return __result + (__last - __first);
    } else {
      return std::transform(__first, __last, __result, __op);
    }
  }
};

template <class _ExecutionPolicy>
struct __transform_binary<__statpart_backend_tag, _ExecutionPolicy> {
  template <class _Policy,
            class _ForwardIterator1,
            class _ForwardIterator2,
            class _ForwardOutIterator,
            class _BinaryOperation>
  _LIBCPP_HIDE_FROM_ABI optional<_ForwardOutIterator>
  operator()(_Policy&&,
             _ForwardIterator1 __first1,
             _ForwardIterator1 __last1,
             _ForwardIterator2 __first2,
             _ForwardOutIterator __result,
             _BinaryOperation __op) const noexcept {
    if constexpr (__is_parallel_execution_policy_v<_ExecutionPolicy> &&
                  __has_random_access_iterator_category_or_concept<_ForwardIterator1>::value &&
                  __has_random_access_iterator_category_or_concept<_ForwardIterator2>::value &&
                  __has_random_access_iterator_category_or_concept<_ForwardOutIterator>::value) {
      auto __res = __statpart::__for_each(
          __first1,
          __last1,
          [__op, __first1, __first2, __result](_ForwardIterator1 __brick_first, _ForwardIterator1 __brick_last) {
            return std::transform(
                __brick_first,
                __brick_last,
                __first2 + (__brick_first - __first1),
                __result + (__brick_first - __first1),
                __op);
          });
      if (!__res)
        return nullopt;
      return __result + (__last1 - __first1);
    } else {
      return std::transform(__first1, __last1, __first2, __result, __op);
    }
  }
};

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
