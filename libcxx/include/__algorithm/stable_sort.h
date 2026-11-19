//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___ALGORITHM_STABLE_SORT_H
#define _LIBCPP___ALGORITHM_STABLE_SORT_H

#include <__algorithm/comp.h>
#include <__algorithm/comp_ref_type.h>
#include <__algorithm/inplace_merge.h>
#include <__algorithm/iterator_operations.h>
#include <__algorithm/lower_bound.h>
#include <__algorithm/min.h>
#include <__algorithm/move.h>
#include <__algorithm/radix_sort.h>
#include <__algorithm/rotate.h>
#include <__algorithm/sort.h>
#include <__algorithm/swap_ranges.h>
#include <__algorithm/upper_bound.h>
#include <__assert>
#include <__config>
#include <__cstddef/ptrdiff_t.h>
#include <__cstddef/size_t.h>
#include <__debug_utils/strict_weak_ordering_check.h>
#include <__functional/identity.h>
#include <__iterator/iterator_traits.h>
#include <__memory/construct_at.h>
#include <__memory/destruct_n.h>
#include <__memory/unique_ptr.h>
#include <__memory/unique_temporary_buffer.h>
#include <__type_traits/desugars_to.h>
#include <__type_traits/enable_if.h>
#include <__type_traits/is_constant_evaluated.h>
#include <__type_traits/is_same.h>
#include <__type_traits/is_trivially_assignable.h>
#include <__type_traits/is_trivially_destructible.h>
#include <__utility/move.h>
#include <__utility/pair.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_PUSH_MACROS
#include <__undef_macros>

_LIBCPP_BEGIN_NAMESPACE_STD

// std::stable_sort is a port of WikiSort (https://github.com/BonzaiThePenguin/WikiSort), a merge sort that merges in
// place: it calls the comparison O(N log(N)) times without allocating memory. It sorts short runs with an insertion
// sort and then merges adjacent runs, level by level:
// - Runs that fit into a small buffer are merged through that buffer.
// - Longer runs are merged block by block. The first run is cut into blocks of about sqrt(N) elements, which are rolled
//   through the second run and dropped where they belong. Each block is then merged with the elements that ended up
//   after it, which only takes a buffer as large as one block. Remembering the original order of the blocks, and
//   providing that buffer when the blocks are too large for the small one, takes sqrt(N) elements each, which are
//   borrowed from the range itself: these are elements with distinct values ("keys"), so that they can be put back
//   where they belong once all the runs have been merged.

template <class _AlgPolicy, class _Compare, class _InputIterator1, class _InputIterator2>
_LIBCPP_CONSTEXPR_SINCE_CXX26 void __merge_move_construct(
    _InputIterator1 __first1,
    _InputIterator1 __last1,
    _InputIterator2 __first2,
    _InputIterator2 __last2,
    typename iterator_traits<_InputIterator1>::value_type* __result,
    _Compare __comp) {
  using _Ops = _IterOps<_AlgPolicy>;

  typedef typename iterator_traits<_InputIterator1>::value_type value_type;
  __destruct_n __d(0);
  unique_ptr<value_type, __destruct_n&> __h(__result, __d);
  for (; true; ++__result) {
    if (__first1 == __last1) {
      for (; __first2 != __last2; ++__first2, (void)++__result, __d.template __incr<value_type>())
        std::__construct_at(__result, _Ops::__iter_move(__first2));
      __h.release();
      return;
    }
    if (__first2 == __last2) {
      for (; __first1 != __last1; ++__first1, (void)++__result, __d.template __incr<value_type>())
        std::__construct_at(__result, _Ops::__iter_move(__first1));
      __h.release();
      return;
    }
    if (__comp(*__first2, *__first1)) {
      std::__construct_at(__result, _Ops::__iter_move(__first2));
      __d.template __incr<value_type>();
      ++__first2;
    } else {
      std::__construct_at(__result, _Ops::__iter_move(__first1));
      __d.template __incr<value_type>();
      ++__first1;
    }
  }
}

template <class _AlgPolicy, class _Compare, class _InputIterator1, class _InputIterator2, class _OutputIterator>
_LIBCPP_CONSTEXPR_SINCE_CXX26 void __merge_move_assign(
    _InputIterator1 __first1,
    _InputIterator1 __last1,
    _InputIterator2 __first2,
    _InputIterator2 __last2,
    _OutputIterator __result,
    _Compare __comp) {
  using _Ops = _IterOps<_AlgPolicy>;

  for (; __first1 != __last1; ++__result) {
    if (__first2 == __last2) {
      for (; __first1 != __last1; ++__first1, (void)++__result)
        *__result = _Ops::__iter_move(__first1);
      return;
    }
    if (__comp(*__first2, *__first1)) {
      *__result = _Ops::__iter_move(__first2);
      ++__first2;
    } else {
      *__result = _Ops::__iter_move(__first1);
      ++__first1;
    }
  }
  for (; __first2 != __last2; ++__first2, (void)++__result)
    *__result = _Ops::__iter_move(__first2);
}

// Ranges with at most this many elements are sorted with an insertion sort. 64 is the largest power of two for which
// that takes at most the N log^2(N) comparisons that std::stable_sort is allowed to make.
template <class _Tp>
struct __stable_sort_switch {
  static const unsigned value = is_trivially_copy_assignable<_Tp>::value ? 64 : 7;
};

#if _LIBCPP_STD_VER >= 17
template <class _Tp>
constexpr unsigned __radix_sort_min_bound() {
  static_assert(__is_ordered_integer_representable_v<_Tp>);
  if constexpr (sizeof(_Tp) == 1) {
    return 1 << 8;
  }

  return 1 << 10;
}

template <class _Tp>
constexpr unsigned __radix_sort_max_bound() {
  static_assert(__is_ordered_integer_representable_v<_Tp>);
  if constexpr (sizeof(_Tp) >= 8) {
    return 1 << 15;
  }

  return 1 << 16;
}
#endif // _LIBCPP_STD_VER >= 17

// The buffer that std::stable_sort merges through. WikiSort was tuned with a buffer of 512 elements; this one is also
// limited to 4096 bytes because it lives on the stack.
template <class _Tp>
struct __stable_sort_stack_cache {
  static const size_t __capacity = sizeof(_Tp) > 8 ? 4096 / sizeof(_Tp) : 512;

  // An array can't be empty, unlike the buffer when the elements are too large for it.
  _ALIGNAS_TYPE(_Tp) unsigned char __storage_[__capacity == 0 ? 1 : __capacity * sizeof(_Tp)];

  _LIBCPP_NO_CFI _Tp* __data() _NOEXCEPT { return static_cast<_Tp*>(static_cast<void*>(__storage_)); }
};

// Keeps track of the elements that have been moved into the buffer that std::stable_sort merges through, so that they
// are destroyed if a comparison or a move throws in the meantime.
template <class _AlgPolicy, class _Tp>
struct __stable_sort_cache {
  _Tp* __data_;
  ptrdiff_t __capacity_;
  ptrdiff_t __size_;

  _LIBCPP_CONSTEXPR_SINCE_CXX26 __stable_sort_cache(_Tp* __data, ptrdiff_t __capacity)
      : __data_(__data), __capacity_(__capacity), __size_(0) {}
  __stable_sort_cache(const __stable_sort_cache&)            = delete;
  __stable_sort_cache& operator=(const __stable_sort_cache&) = delete;
  _LIBCPP_CONSTEXPR_SINCE_CXX26 ~__stable_sort_cache() { __clear(); }

  _LIBCPP_CONSTEXPR_SINCE_CXX26 _Tp* __end() const { return __data_ + __size_; }

  template <class _Iter>
  _LIBCPP_CONSTEXPR_SINCE_CXX26 void __move_in(_Iter __first, _Iter __last) {
    for (; __first != __last; ++__first, (void)++__size_)
      std::__construct_at(__data_ + __size_, _IterOps<_AlgPolicy>::__iter_move(__first));
  }

  _LIBCPP_CONSTEXPR_SINCE_CXX26 void __clear() _NOEXCEPT {
    if (!is_trivially_destructible<_Tp>::value || __libcpp_is_constant_evaluated()) {
      while (__size_ != 0)
        std::__destroy_at(__data_ + --__size_);
    }
    __size_ = 0;
  }
};

// Generates the runs that a bottom-up merge sort of __size elements merges on each of its levels. The length of the
// runs is scaled by a fraction rather than doubled from one level to the next, so that the runs of a level never
// differ by more than one element whatever __size is.
template <class _Diff>
struct __stable_sort_runs {
  _Diff __size_;
  _Diff __denominator_;
  _Diff __run_length_;
  _Diff __remainder_step_;
  _Diff __position_;
  _Diff __remainder_;

  _LIBCPP_CONSTEXPR_SINCE_CXX26 __stable_sort_runs(_Diff __size, _Diff __min_run_length)
      : __size_(__size), __denominator_(1), __run_length_(0), __remainder_step_(0), __position_(0), __remainder_(0) {
    _Diff __power_of_two = 1;
    while (__power_of_two <= __size / 2)
      __power_of_two += __power_of_two;
    __denominator_    = __power_of_two / __min_run_length;
    __run_length_     = __size / __denominator_;
    __remainder_step_ = __size % __denominator_;
  }

  _LIBCPP_CONSTEXPR_SINCE_CXX26 void __restart() {
    __position_  = 0;
    __remainder_ = 0;
  }

  _LIBCPP_CONSTEXPR_SINCE_CXX26 bool __at_end() const { return !(__position_ < __size_); }

  // Moves past the next run and returns the offset of its end, which is __position_ from then on.
  _LIBCPP_CONSTEXPR_SINCE_CXX26 _Diff __next() {
    __position_ += __run_length_;
    __remainder_ += __remainder_step_;
    if (!(__remainder_ < __denominator_)) {
      __remainder_ -= __denominator_;
      ++__position_;
    }
    return __position_;
  }

  // Makes every two consecutive runs a single run, and returns whether there is more than one run left.
  _LIBCPP_CONSTEXPR_SINCE_CXX26 bool __next_level() {
    __run_length_ += __run_length_;
    __remainder_step_ += __remainder_step_;
    if (!(__remainder_step_ < __denominator_)) {
      __remainder_step_ -= __denominator_;
      ++__run_length_;
    }
    return __run_length_ < __size_;
  }
};

// Returns the largest integer whose square is at most __n, which has to be positive.
template <class _Diff>
_LIBCPP_CONSTEXPR_SINCE_CXX26 _Diff __stable_sort_isqrt(_Diff __n) {
  _Diff __low  = 1;
  _Diff __high = __n / 2 + 1;
  while (__high - __low > 1) {
    _Diff __mid = __low + (__high - __low) / 2;
    if (__mid <= __n / __mid)
      __low = __mid;
    else
      __high = __mid;
  }
  return __low;
}

// The three functions below return the same iterator as lower_bound or upper_bound. They start by looking at every
// __skip-th element from one end of the range, because that is where the result is expected to be when the range has
// about __unique distinct values.
template <class _Diff>
_LIBCPP_CONSTEXPR_SINCE_CXX26 _Diff __stable_sort_skip(_Diff __size, _Diff __unique) {
  if (__unique < 1 || __size / __unique < 1)
    return _Diff(1);
  return __size / __unique;
}

template <class _AlgPolicy, class _Compare, class _RandomAccessIterator, class _Tp>
_LIBCPP_CONSTEXPR_SINCE_CXX26 _RandomAccessIterator __stable_sort_find_first_forward(
    _RandomAccessIterator __first,
    _RandomAccessIterator __last,
    const _Tp& __value,
    _Compare __comp,
    typename iterator_traits<_RandomAccessIterator>::difference_type __unique) {
  typedef typename iterator_traits<_RandomAccessIterator>::difference_type difference_type;
  __identity __proj;
  if (__first == __last)
    return __first;

  const difference_type __skip  = std::__stable_sort_skip<difference_type>(__last - __first, __unique);
  _RandomAccessIterator __index = __first + __skip;
  while (__comp(*(__index - difference_type(1)), __value)) {
    if (__last - __index <= __skip)
      return std::__lower_bound<_AlgPolicy>(__index, __last, __value, __comp, __proj);
    __index += __skip;
  }
  return std::__lower_bound<_AlgPolicy>(__index - __skip, __index, __value, __comp, __proj);
}

template <class _AlgPolicy, class _Compare, class _RandomAccessIterator, class _Tp>
_LIBCPP_CONSTEXPR_SINCE_CXX26 _RandomAccessIterator __stable_sort_find_last_forward(
    _RandomAccessIterator __first,
    _RandomAccessIterator __last,
    const _Tp& __value,
    _Compare __comp,
    typename iterator_traits<_RandomAccessIterator>::difference_type __unique) {
  typedef typename iterator_traits<_RandomAccessIterator>::difference_type difference_type;
  __identity __proj;
  if (__first == __last)
    return __first;

  const difference_type __skip  = std::__stable_sort_skip<difference_type>(__last - __first, __unique);
  _RandomAccessIterator __index = __first + __skip;
  while (!__comp(__value, *(__index - difference_type(1)))) {
    if (__last - __index <= __skip)
      return std::__upper_bound<_AlgPolicy>(__index, __last, __value, __comp, __proj);
    __index += __skip;
  }
  return std::__upper_bound<_AlgPolicy>(__index - __skip, __index, __value, __comp, __proj);
}

template <class _AlgPolicy, class _Compare, class _RandomAccessIterator, class _Tp>
_LIBCPP_CONSTEXPR_SINCE_CXX26 _RandomAccessIterator __stable_sort_find_first_backward(
    _RandomAccessIterator __first,
    _RandomAccessIterator __last,
    const _Tp& __value,
    _Compare __comp,
    typename iterator_traits<_RandomAccessIterator>::difference_type __unique) {
  typedef typename iterator_traits<_RandomAccessIterator>::difference_type difference_type;
  __identity __proj;
  if (__first == __last)
    return __first;

  const difference_type __skip  = std::__stable_sort_skip<difference_type>(__last - __first, __unique);
  _RandomAccessIterator __index = __last - __skip;
  while (__index != __first && !__comp(*(__index - difference_type(1)), __value)) {
    if (__index - __first < __skip)
      return std::__lower_bound<_AlgPolicy>(__first, __index, __value, __comp, __proj);
    __index -= __skip;
  }
  return std::__lower_bound<_AlgPolicy>(__index, __index + __skip, __value, __comp, __proj);
}

// Counts the distinct values at the beginning of the sorted range [__first, __last), which mustn't be empty, up to
// __wanted of them. Returns that count and the position of the first element with the last value that was counted.
template <class _AlgPolicy, class _Compare, class _RandomAccessIterator>
_LIBCPP_CONSTEXPR_SINCE_CXX26 pair<__iterator_difference_type<_RandomAccessIterator>, _RandomAccessIterator>
__stable_sort_count_keys(_RandomAccessIterator __first,
                         _RandomAccessIterator __last,
                         typename iterator_traits<_RandomAccessIterator>::difference_type __wanted,
                         _Compare __comp) {
  typedef typename iterator_traits<_RandomAccessIterator>::difference_type difference_type;

  difference_type __count = 1;
  while (__count < __wanted) {
    _RandomAccessIterator __next = std::__stable_sort_find_last_forward<_AlgPolicy, _Compare>(
        __first + difference_type(1), __last, *__first, __comp, __wanted - __count);
    if (__next == __last)
      break;
    __first = __next;
    ++__count;
  }
  return pair<difference_type, _RandomAccessIterator>(__count, __first);
}

// Moves the first element with each of the __count values found by __stable_sort_count_keys, the last of which is at
// __key, to the beginning of the range that starts at __first, without changing the order of the other elements.
template <class _AlgPolicy, class _Compare, class _RandomAccessIterator>
_LIBCPP_CONSTEXPR_SINCE_CXX26 void __stable_sort_pull_keys(
    _RandomAccessIterator __first,
    _RandomAccessIterator __key,
    typename iterator_traits<_RandomAccessIterator>::difference_type __count,
    _Compare __comp) {
  typedef typename iterator_traits<_RandomAccessIterator>::difference_type difference_type;

  // The keys gathered so far are the __pulled elements that start at __key. They move towards __first one value at a
  // time, jumping over the other elements with the value that comes before theirs.
  difference_type __pulled = 1;
  for (; __pulled < __count && __key != __first; ++__pulled) {
    _RandomAccessIterator __previous = std::__stable_sort_find_first_backward<_AlgPolicy, _Compare>(
        __first, __key, *(__key - difference_type(1)), __comp, __count - __pulled);
    // The rest of this function only matters when the comparison isn't a strict weak order: the keys still have to end
    // up inside of the range.
    if (__previous == __key)
      --__previous;
    std::__rotate<_AlgPolicy>(__previous + difference_type(1), __key, __key + __pulled);
    __key = __previous;
  }
  std::__rotate<_AlgPolicy>(__first, __key, __key + __pulled);
}

// Merges the sorted keys in [__first, __keys_last) back into the sorted range [__keys_last, __last) they were pulled
// out of. A key goes before the elements that are equivalent to it, since it was the first of them.
template <class _AlgPolicy, class _Compare, class _RandomAccessIterator>
_LIBCPP_CONSTEXPR_SINCE_CXX26 void __stable_sort_redistribute_keys(
    _RandomAccessIterator __first, _RandomAccessIterator __keys_last, _RandomAccessIterator __last, _Compare __comp) {
  typedef typename iterator_traits<_RandomAccessIterator>::difference_type difference_type;

  difference_type __unique = (__keys_last - __first) + (__keys_last - __first);
  for (; __first != __keys_last; __unique -= 2) {
    _RandomAccessIterator __position =
        std::__stable_sort_find_first_forward<_AlgPolicy, _Compare>(__keys_last, __last, *__first, __comp, __unique);
    difference_type __amount = __position - __keys_last;
    std::__rotate<_AlgPolicy>(__first, __keys_last, __position);
    __first += __amount + difference_type(1);
    __keys_last += __amount;
  }
}

// Merges [__first, __middle) and [__middle, __last) when the elements of the first range have been swapped with the
// same number of elements starting at __buffer. The elements that were at __buffer end up there again, in any order.
template <class _AlgPolicy, class _Compare, class _RandomAccessIterator>
_LIBCPP_CONSTEXPR_SINCE_CXX26 void __stable_sort_merge_internal(
    _RandomAccessIterator __first,
    _RandomAccessIterator __middle,
    _RandomAccessIterator __last,
    _RandomAccessIterator __buffer,
    _Compare __comp) {
  using _Ops = _IterOps<_AlgPolicy>;

  _RandomAccessIterator __buffer_last = __buffer + (__middle - __first);
  for (; __buffer != __buffer_last && __middle != __last; ++__first) {
    if (__comp(*__middle, *__buffer)) {
      _Ops::iter_swap(__first, __middle);
      ++__middle;
    } else {
      _Ops::iter_swap(__first, __buffer);
      ++__buffer;
    }
  }
  std::__swap_ranges<_AlgPolicy>(__buffer, __buffer_last, __first);
}

// Merges [__first, __middle) and [__middle, __last) without a buffer, by rotating the elements of the first range into
// place one run of equivalent elements at a time. This is only efficient when the first range has few distinct values,
// which is the case when not enough keys could be found to provide a buffer.
template <class _AlgPolicy, class _Compare, class _RandomAccessIterator>
_LIBCPP_CONSTEXPR_SINCE_CXX26 void __stable_sort_merge_in_place(
    _RandomAccessIterator __first, _RandomAccessIterator __middle, _RandomAccessIterator __last, _Compare __comp) {
  typedef typename iterator_traits<_RandomAccessIterator>::difference_type difference_type;
  __identity __proj;

  while (__first != __middle && __middle != __last) {
    _RandomAccessIterator __position = std::__lower_bound<_AlgPolicy>(__middle, __last, *__first, __comp, __proj);
    difference_type __amount         = __position - __middle;
    std::__rotate<_AlgPolicy>(__first, __middle, __position);
    __first += __amount;
    __middle = __position;
    if (__middle == __last)
      break;
    // Searching from the second element on makes sure that this loop terminates even if the comparison isn't a strict
    // weak order.
    __first = std::__upper_bound<_AlgPolicy>(__first + difference_type(1), __middle, *__first, __comp, __proj);
  }
}

// Not a scoped enumeration, because the enumerators of those can't be named in C++03.
enum __stable_sort_stash {
  __stable_sort_stash_in_place,
  __stable_sort_stash_in_cache,
  __stable_sort_stash_in_swap_buffer
};

// Moves the block [__first, __last) to where it can be merged from most efficiently.
template <class _AlgPolicy, class _RandomAccessIterator>
_LIBCPP_CONSTEXPR_SINCE_CXX26 __stable_sort_stash __stable_sort_stash_block(
    _RandomAccessIterator __first,
    _RandomAccessIterator __last,
    _RandomAccessIterator __swap_buffer,
    bool __has_swap_buffer,
    __stable_sort_cache<_AlgPolicy, typename iterator_traits<_RandomAccessIterator>::value_type>& __cache) {
  if (__last - __first <= __cache.__capacity_) {
    __cache.__move_in(__first, __last);
    return __stable_sort_stash_in_cache;
  }
  if (__has_swap_buffer) {
    std::__swap_ranges<_AlgPolicy>(__first, __last, __swap_buffer);
    return __stable_sort_stash_in_swap_buffer;
  }
  return __stable_sort_stash_in_place;
}

// Merges the block that belongs in [__first, __middle) and was stashed by __stable_sort_stash_block with
// [__middle, __last).
template <class _AlgPolicy, class _Compare, class _RandomAccessIterator>
_LIBCPP_CONSTEXPR_SINCE_CXX26 void __stable_sort_merge_stashed_block(
    _RandomAccessIterator __first,
    _RandomAccessIterator __middle,
    _RandomAccessIterator __last,
    _Compare __comp,
    __stable_sort_stash __stash,
    _RandomAccessIterator __swap_buffer,
    __stable_sort_cache<_AlgPolicy, typename iterator_traits<_RandomAccessIterator>::value_type>& __cache) {
  switch (__stash) {
  case __stable_sort_stash_in_cache:
    std::__half_inplace_merge<_AlgPolicy>(__cache.__data_, __cache.__end(), __middle, __last, __first, __comp);
    __cache.__clear();
    break;
  case __stable_sort_stash_in_swap_buffer:
    std::__stable_sort_merge_internal<_AlgPolicy, _Compare>(__first, __middle, __last, __swap_buffer, __comp);
    break;
  case __stable_sort_stash_in_place:
    std::__stable_sort_merge_in_place<_AlgPolicy, _Compare>(__first, __middle, __last, __comp);
    break;
  }
}

// Merges the sorted ranges A = [__first, __middle) and B = [__middle, __last), neither of which is empty, block by
// block. [__keys, ...) holds at least as many sorted keys as A has blocks of __block_size elements, and, if
// __has_swap_buffer, [__swap_buffer, ...) holds at least __block_size elements whose order doesn't matter.
template <class _AlgPolicy, class _Compare, class _RandomAccessIterator>
_LIBCPP_CONSTEXPR_SINCE_CXX26 void __stable_sort_block_merge(
    _RandomAccessIterator __first,
    _RandomAccessIterator __middle,
    _RandomAccessIterator __last,
    _Compare __comp,
    typename iterator_traits<_RandomAccessIterator>::difference_type __block_size,
    _RandomAccessIterator __keys,
    _RandomAccessIterator __swap_buffer,
    bool __has_swap_buffer,
    __stable_sort_cache<_AlgPolicy, typename iterator_traits<_RandomAccessIterator>::value_type>& __cache) {
  using _Ops = _IterOps<_AlgPolicy>;
  typedef typename iterator_traits<_RandomAccessIterator>::difference_type difference_type;
  __identity __proj;

  // Only the first block of A may have fewer than __block_size elements. The other ones are the blocks that roll
  // through B: they stay next to each other, but not in order.
  _RandomAccessIterator __blocks      = __first + (__middle - __first) % __block_size;
  _RandomAccessIterator __blocks_last = __middle;

  // The first element of every rolling block is swapped with a key. Since the keys are sorted, comparing the first
  // elements of two blocks then tells which of them came first, and swapping the smallest remaining key back restores
  // both the block and the keys.
  _RandomAccessIterator __key = __keys;
  for (_RandomAccessIterator __block = __blocks; __block != __blocks_last; __block += __block_size, (void)++__key)
    _Ops::iter_swap(__block, __key);
  __key = __keys;

  // The block of A that was dropped last still has to be merged with the elements of B that follow it, up to where the
  // next block gets dropped.
  _RandomAccessIterator __dropped      = __first;
  _RandomAccessIterator __dropped_last = __blocks;
  __stable_sort_stash __stash =
      std::__stable_sort_stash_block<_AlgPolicy>(__dropped, __dropped_last, __swap_buffer, __has_swap_buffer, __cache);

  // [__last_b, __blocks) is the block of B that the rolling blocks went past last, or what is left of it.
  _RandomAccessIterator __last_b = __blocks;
  difference_type __next_b_size  = std::min<difference_type>(__block_size, __last - __middle);

  while (__blocks != __blocks_last) {
    if ((__last_b != __blocks && !__comp(*(__blocks - difference_type(1)), *__key)) || __next_b_size == 0) {
      // The first element of the first remaining block of A, which is where the smallest remaining key is, belongs
      // before the end of the last block of B: that block of A gets dropped there.
      _RandomAccessIterator __b_split = std::__lower_bound<_AlgPolicy>(__last_b, __blocks, *__key, __comp, __proj);
      difference_type __b_remaining   = __blocks - __b_split;

      _RandomAccessIterator __min_block = __blocks;
      for (_RandomAccessIterator __block = __blocks + __block_size; __block != __blocks_last; __block += __block_size)
        if (__comp(*__block, *__min_block))
          __min_block = __block;
      _RandomAccessIterator __next_blocks = __blocks + __block_size;
      if (__min_block != __blocks)
        std::__swap_ranges<_AlgPolicy>(__blocks, __next_blocks, __min_block);
      _Ops::iter_swap(__blocks, __key);
      ++__key;

      std::__stable_sort_merge_stashed_block<_AlgPolicy, _Compare>(
          __dropped, __dropped_last, __b_split, __comp, __stash, __swap_buffer, __cache);

      // The block moves in front of the elements of B that come after it. When it is stashed away, those elements only
      // have to be moved to the end of the place it leaves behind, which is cheaper than a rotation.
      __stash = std::__stable_sort_stash_block<_AlgPolicy>(
          __blocks, __next_blocks, __swap_buffer, __has_swap_buffer, __cache);
      __dropped      = __b_split;
      __dropped_last = __next_blocks - __b_remaining;
      switch (__stash) {
      case __stable_sort_stash_in_cache:
        std::__move<_AlgPolicy>(__b_split, __blocks, __dropped_last);
        break;
      case __stable_sort_stash_in_swap_buffer:
        std::__swap_ranges<_AlgPolicy>(__b_split, __blocks, __dropped_last);
        break;
      case __stable_sort_stash_in_place:
        std::__rotate<_AlgPolicy>(__b_split, __blocks, __next_blocks);
        break;
      }
      __last_b = __dropped_last;
      __blocks = __next_blocks;
    } else if (__next_b_size < __block_size) {
      // The last block of B is too short to be swapped with a block of A.
      std::__rotate<_AlgPolicy>(__blocks, __blocks_last, __blocks_last + __next_b_size);
      __last_b = __blocks;
      __blocks += __next_b_size;
      __blocks_last += __next_b_size;
      __next_b_size = 0;
    } else {
      std::__swap_ranges<_AlgPolicy>(__blocks, __blocks + __block_size, __blocks_last);
      __last_b = __blocks;
      __blocks += __block_size;
      __blocks_last += __block_size;
      __next_b_size = std::min<difference_type>(__block_size, __last - __blocks_last);
    }
  }

  std::__stable_sort_merge_stashed_block<_AlgPolicy, _Compare>(
      __dropped, __dropped_last, __last, __comp, __stash, __swap_buffer, __cache);
}

// Takes care of the runs [__first, __middle) and [__middle, __last) when they don't have to be merged element by
// element, and returns whether they do.
template <class _AlgPolicy, class _Compare, class _RandomAccessIterator>
_LIBCPP_CONSTEXPR_SINCE_CXX26 bool __stable_sort_needs_merge(
    _RandomAccessIterator __first, _RandomAccessIterator __middle, _RandomAccessIterator __last, _Compare __comp) {
  typedef typename iterator_traits<_RandomAccessIterator>::difference_type difference_type;

  if (__comp(*(__last - difference_type(1)), *__first)) {
    std::__rotate<_AlgPolicy>(__first, __middle, __last);
    return false;
  }
  return __comp(*__middle, *(__middle - difference_type(1))) ? true : false;
}

// Merges every pair of runs, each of which fits into the cache.
template <class _AlgPolicy, class _Compare, class _RandomAccessIterator>
_LIBCPP_CONSTEXPR_SINCE_CXX26 void __stable_sort_merge_runs_through_cache(
    _RandomAccessIterator __first,
    __stable_sort_runs<typename iterator_traits<_RandomAccessIterator>::difference_type>& __runs,
    _Compare __comp,
    __stable_sort_cache<_AlgPolicy, typename iterator_traits<_RandomAccessIterator>::value_type>& __cache) {
  for (__runs.__restart(); !__runs.__at_end();) {
    _RandomAccessIterator __a_first = __first + __runs.__position_;
    _RandomAccessIterator __a_last  = __first + __runs.__next();
    _RandomAccessIterator __b_last  = __first + __runs.__next();
    if (std::__stable_sort_needs_merge<_AlgPolicy, _Compare>(__a_first, __a_last, __b_last, __comp))
      std::__buffered_inplace_merge<_AlgPolicy>(
          __a_first, __a_last, __b_last, __comp, __a_last - __a_first, __b_last - __a_last, __cache.__data_);
  }
}

// Merges every four runs, which fit into the cache together, into one. Merging each pair of runs into the cache and
// the two results back into the range takes the place of two levels of the merge sort, and moves each element twice
// instead of three times.
template <class _AlgPolicy, class _Compare, class _RandomAccessIterator>
_LIBCPP_CONSTEXPR_SINCE_CXX26 void __stable_sort_merge_runs_through_cache_twice(
    _RandomAccessIterator __first,
    __stable_sort_runs<typename iterator_traits<_RandomAccessIterator>::difference_type>& __runs,
    _Compare __comp,
    __stable_sort_cache<_AlgPolicy, typename iterator_traits<_RandomAccessIterator>::value_type>& __cache) {
  typedef typename iterator_traits<_RandomAccessIterator>::value_type value_type;
  typedef typename iterator_traits<_RandomAccessIterator>::difference_type difference_type;

  for (__runs.__restart(); !__runs.__at_end();) {
    _RandomAccessIterator __a1   = __first + __runs.__position_;
    _RandomAccessIterator __b1   = __first + __runs.__next();
    _RandomAccessIterator __a2   = __first + __runs.__next();
    _RandomAccessIterator __b2   = __first + __runs.__next();
    _RandomAccessIterator __last = __first + __runs.__next();

    if (__comp(*(__a2 - difference_type(1)), *__a1)) {
      __cache.__move_in(__b1, __a2);
      __cache.__move_in(__a1, __b1);
    } else if (__comp(*__b1, *(__b1 - difference_type(1)))) {
      std::__merge_move_construct<_AlgPolicy, _Compare>(__a1, __b1, __b1, __a2, __cache.__end(), __comp);
      __cache.__size_ += static_cast<ptrdiff_t>(__a2 - __a1);
    } else {
      if (!__comp(*__b2, *(__b2 - difference_type(1))) && !__comp(*__a2, *(__a2 - difference_type(1))))
        continue;
      __cache.__move_in(__a1, __a2);
    }
    value_type* __cache_middle = __cache.__end();

    if (__comp(*(__last - difference_type(1)), *__a2)) {
      __cache.__move_in(__b2, __last);
      __cache.__move_in(__a2, __b2);
    } else if (__comp(*__b2, *(__b2 - difference_type(1)))) {
      std::__merge_move_construct<_AlgPolicy, _Compare>(__a2, __b2, __b2, __last, __cache.__end(), __comp);
      __cache.__size_ += static_cast<ptrdiff_t>(__last - __a2);
    } else {
      __cache.__move_in(__a2, __last);
    }

    if (__comp(*(__cache.__end() - 1), *__cache.__data_))
      std::__move<_AlgPolicy>(
          __cache.__data_, __cache_middle, std::__move<_AlgPolicy>(__cache_middle, __cache.__end(), __a1).__out_);
    else if (__comp(*__cache_middle, *(__cache_middle - 1)))
      std::__merge_move_assign<_AlgPolicy, _Compare>(
          __cache.__data_, __cache_middle, __cache_middle, __cache.__end(), __a1, __comp);
    else
      std::__move<_AlgPolicy>(__cache.__data_, __cache.__end(), __a1);
    __cache.__clear();
  }
}

// Merges every pair of runs that are too long for the cache, but so short that merging them block by block would
// spend more time looking for keys than merging.
template <class _AlgPolicy, class _Compare, class _RandomAccessIterator>
_LIBCPP_CONSTEXPR_SINCE_CXX26 void __stable_sort_merge_short_runs_in_place(
    _RandomAccessIterator __first,
    __stable_sort_runs<typename iterator_traits<_RandomAccessIterator>::difference_type>& __runs,
    _Compare __comp,
    __stable_sort_cache<_AlgPolicy, typename iterator_traits<_RandomAccessIterator>::value_type>& __cache) {
  for (__runs.__restart(); !__runs.__at_end();) {
    _RandomAccessIterator __a_first = __first + __runs.__position_;
    _RandomAccessIterator __a_last  = __first + __runs.__next();
    _RandomAccessIterator __b_last  = __first + __runs.__next();
    if (std::__stable_sort_needs_merge<_AlgPolicy, _Compare>(__a_first, __a_last, __b_last, __comp))
      std::__inplace_merge<_AlgPolicy>(
          __a_first,
          __a_last,
          __b_last,
          __comp,
          __a_last - __a_first,
          __b_last - __a_last,
          __cache.__data_,
          __cache.__capacity_);
  }
}

// Merges every pair of runs block by block.
template <class _AlgPolicy, class _Compare, class _RandomAccessIterator>
_LIBCPP_CONSTEXPR_SINCE_CXX26 void __stable_sort_merge_runs_in_place(
    _RandomAccessIterator __first,
    __stable_sort_runs<typename iterator_traits<_RandomAccessIterator>::difference_type>& __runs,
    _Compare __comp,
    __stable_sort_cache<_AlgPolicy, typename iterator_traits<_RandomAccessIterator>::value_type>& __cache) {
  typedef typename iterator_traits<_RandomAccessIterator>::difference_type difference_type;

  const difference_type __run_length = __runs.__run_length_;
  difference_type __block_size       = std::__stable_sort_isqrt(__run_length);
  difference_type __buffer_size      = __run_length / __block_size + difference_type(1);

  // Blocks that don't fit into the cache are merged through a second buffer of keys.
  const bool __wants_swap_buffer = __block_size > __cache.__capacity_;
  const difference_type __wanted = __wants_swap_buffer ? __buffer_size + __buffer_size : __buffer_size;

  // The keys are pulled out of the first run of a pair that has enough of them or, failing that, as many as possible.
  // They are only looked for once per level since all the pairs of runs can share them. WikiSort also looks for them
  // in the second run of each pair, and for the two buffers separately, but neither is needed to merge efficiently:
  // without enough keys, the first runs have so few distinct values that their blocks are quickly merged in place.
  _RandomAccessIterator __keys           = __first;
  _RandomAccessIterator __last_key       = __first;
  _RandomAccessIterator __keys_pair_last = __first;
  difference_type __key_count            = 0;
  for (__runs.__restart(); __key_count != __wanted && !__runs.__at_end();) {
    _RandomAccessIterator __a_first = __first + __runs.__position_;
    _RandomAccessIterator __a_last  = __first + __runs.__next();
    _RandomAccessIterator __b_last  = __first + __runs.__next();
    pair<difference_type, _RandomAccessIterator> __found =
        std::__stable_sort_count_keys<_AlgPolicy, _Compare>(__a_first, __a_last, __wanted, __comp);
    if (__found.first > __key_count) {
      __key_count      = __found.first;
      __last_key       = __found.second;
      __keys           = __a_first;
      __keys_pair_last = __b_last;
    }
  }
  std::__stable_sort_pull_keys<_AlgPolicy, _Compare>(__keys, __last_key, __key_count, __comp);

  // With fewer keys than wanted, the blocks have to be larger for the keys to be able to tag all of them.
  const bool __has_swap_buffer = __wants_swap_buffer && __key_count == __wanted;
  if (!__has_swap_buffer)
    __buffer_size = __key_count;
  __block_size                        = __run_length / __buffer_size + difference_type(1);
  _RandomAccessIterator __swap_buffer = __keys + __buffer_size;

  for (__runs.__restart(); !__runs.__at_end();) {
    _RandomAccessIterator __a_first = __first + __runs.__position_;
    _RandomAccessIterator __a_last  = __first + __runs.__next();
    _RandomAccessIterator __b_last  = __first + __runs.__next();
    if (__a_first == __keys) {
      __a_first += __key_count;
      // At most 2 * (__run_length / sqrt(__run_length) + 1) keys are pulled out. That is fewer than a run has
      // elements, because runs are only merged block by block when they have at least 16 of them.
      _LIBCPP_ASSERT_INTERNAL(__a_first != __a_last, "the keys can't take up a whole run");
    }
    if (std::__stable_sort_needs_merge<_AlgPolicy, _Compare>(__a_first, __a_last, __b_last, __comp))
      std::__stable_sort_block_merge<_AlgPolicy, _Compare>(
          __a_first, __a_last, __b_last, __comp, __block_size, __keys, __swap_buffer, __has_swap_buffer, __cache);
  }

  // Only the keys that the blocks were merged through have lost their order.
  if (__has_swap_buffer)
    std::__insertion_sort<_AlgPolicy, _Compare>(__swap_buffer, __keys + __key_count, __comp);
  std::__stable_sort_redistribute_keys<_AlgPolicy, _Compare>(__keys, __keys + __key_count, __keys_pair_last, __comp);
}

template <class _AlgPolicy, class _Compare, class _RandomAccessIterator>
_LIBCPP_CONSTEXPR_SINCE_CXX26 void __stable_sort(
    _RandomAccessIterator __first,
    _RandomAccessIterator __last,
    _Compare __comp,
    typename iterator_traits<_RandomAccessIterator>::difference_type __len,
    typename iterator_traits<_RandomAccessIterator>::value_type* __buff,
    ptrdiff_t __buff_size) {
  typedef typename iterator_traits<_RandomAccessIterator>::value_type value_type;
  typedef typename iterator_traits<_RandomAccessIterator>::difference_type difference_type;

  if (__len <= static_cast<difference_type>(__stable_sort_switch<value_type>::value)) {
    std::__insertion_sort<_AlgPolicy, _Compare>(__first, __last, __comp);
    return;
  }

#if _LIBCPP_STD_VER >= 17
  constexpr auto __default_comp = __desugars_to_v<__less_tag, _Compare, value_type, value_type >;
  constexpr auto __radix_sortable =
      __is_ordered_integer_representable_v<value_type> &&
      is_same_v< value_type&, __iterator_reference<_RandomAccessIterator>>;
  if constexpr (__default_comp && __radix_sortable) {
    if (__len <= __buff_size && __len >= static_cast<difference_type>(std::__radix_sort_min_bound<value_type>()) &&
        __len <= static_cast<difference_type>(std::__radix_sort_max_bound<value_type>())) {
      if (__libcpp_is_constant_evaluated()) {
        for (auto* __p = __buff; __p < __buff + __buff_size; ++__p) {
          std::__construct_at(__p);
        }
      }

      std::__radix_sort(__first, __last, __buff);
      return;
    }
  }
#endif // _LIBCPP_STD_VER >= 17

  __stable_sort_runs<difference_type> __runs(__len, difference_type(4));
  while (!__runs.__at_end()) {
    _RandomAccessIterator __run_first = __first + __runs.__position_;
    std::__insertion_sort<_AlgPolicy, _Compare>(__run_first, __first + __runs.__next(), __comp);
  }

  __stable_sort_cache<_AlgPolicy, value_type> __cache(__buff, __buff_size);
  do {
    const difference_type __run_length = __runs.__run_length_;
    if (__run_length < __cache.__capacity_ / 4 && __run_length <= __len / 4) {
      std::__stable_sort_merge_runs_through_cache_twice<_AlgPolicy, _Compare>(__first, __runs, __comp, __cache);
      __runs.__next_level();
    } else if (__run_length < __cache.__capacity_) {
      std::__stable_sort_merge_runs_through_cache<_AlgPolicy, _Compare>(__first, __runs, __comp, __cache);
    } else if (__run_length < difference_type(16)) {
      std::__stable_sort_merge_short_runs_in_place<_AlgPolicy, _Compare>(__first, __runs, __comp, __cache);
    } else {
      std::__stable_sort_merge_runs_in_place<_AlgPolicy, _Compare>(__first, __runs, __comp, __cache);
    }
  } while (__runs.__next_level());
}

template <class _AlgPolicy, class _RandomAccessIterator, class _Compare>
_LIBCPP_CONSTEXPR_SINCE_CXX26 void
__stable_sort_impl(_RandomAccessIterator __first, _RandomAccessIterator __last, _Compare& __comp) {
  using value_type      = typename iterator_traits<_RandomAccessIterator>::value_type;
  using difference_type = typename iterator_traits<_RandomAccessIterator>::difference_type;

  difference_type __len      = __last - __first;
  const ptrdiff_t __capacity = static_cast<ptrdiff_t>(__stable_sort_stack_cache<value_type>::__capacity);
  if (__capacity == 0 || __len <= static_cast<difference_type>(__stable_sort_switch<value_type>::value)) {
    std::__stable_sort<_AlgPolicy, __comp_ref_type<_Compare> >(
        __first, __last, __comp, __len, static_cast<value_type*>(nullptr), 0);
  } else if (__libcpp_is_constant_evaluated()) {
    // The buffer can't be on the stack during constant evaluation, where raw storage can't hold objects of other
    // types.
    __unique_temporary_buffer<value_type> __unique_buf =
        std::__allocate_unique_temporary_buffer<value_type>(__capacity);
    std::__stable_sort<_AlgPolicy, __comp_ref_type<_Compare> >(
        __first, __last, __comp, __len, __unique_buf.get(), __unique_buf.get_deleter().__count_);
  } else {
    __stable_sort_stack_cache<value_type> __stack_cache;
    std::__stable_sort<_AlgPolicy, __comp_ref_type<_Compare> >(
        __first, __last, __comp, __len, __stack_cache.__data(), __capacity);
  }
  std::__check_strict_weak_ordering_sorted(__first, __last, __comp);
}

template <class _RandomAccessIterator, class _Compare>
_LIBCPP_CONSTEXPR_SINCE_CXX26 void
stable_sort(_RandomAccessIterator __first, _RandomAccessIterator __last, _Compare __comp) {
  std::__stable_sort_impl<_ClassicAlgPolicy>(std::move(__first), std::move(__last), __comp);
}

template <class _RandomAccessIterator>
_LIBCPP_CONSTEXPR_SINCE_CXX26 void stable_sort(_RandomAccessIterator __first, _RandomAccessIterator __last) {
  std::stable_sort(__first, __last, __less<>());
}

_LIBCPP_END_NAMESPACE_STD
_LIBCPP_POP_MACROS

#endif // _LIBCPP___ALGORITHM_STABLE_SORT_H
