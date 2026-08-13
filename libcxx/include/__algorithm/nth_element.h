//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___ALGORITHM_NTH_ELEMENT_H
#define _LIBCPP___ALGORITHM_NTH_ELEMENT_H

#include <__algorithm/comp.h>
#include <__algorithm/comp_ref_type.h>
#include <__algorithm/iterator_operations.h>
#include <__algorithm/sort.h>
#include <__assert>
#include <__config>
#include <__debug_utils/randomize_range.h>
#include <__iterator/iterator_traits.h>
#include <__type_traits/enable_if.h>
#include <__utility/move.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_PUSH_MACROS
#include <__undef_macros>

_LIBCPP_BEGIN_NAMESPACE_STD

// Selects the median of 5 elements and partitions the other 4 around it, so the nth_element
// postconditions hold for nth == first + 2. Uses 6 comparisons, which is optimal
// (Knuth, TAOCP 5.3.3).
template <class _AlgPolicy,
          class _Compare,
          class _RandomAccessIterator,
          __enable_if_t<!__use_branchless_sort<_Compare, _RandomAccessIterator>, int> = 0>
_LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX14 void
__nth_element_median5(_RandomAccessIterator __first, _Compare __comp) {
  using _Ops = _IterOps<_AlgPolicy>;
  typedef typename iterator_traits<_RandomAccessIterator>::difference_type difference_type;

  _RandomAccessIterator __x1 = __first;
  _RandomAccessIterator __x2 = __first + difference_type(1);
  _RandomAccessIterator __x3 = __first + difference_type(2);
  _RandomAccessIterator __x4 = __first + difference_type(3);
  _RandomAccessIterator __x5 = __first + difference_type(4);

  // Sort pairs (__x1, __x2) and (__x3, __x4).
  if (__comp(*__x2, *__x1))
    _Ops::iter_swap(__x1, __x2);
  if (__comp(*__x4, *__x3))
    _Ops::iter_swap(__x3, __x4);

  // Order those pairs by their larger element: *__x1 <= *__x2 <= *__x4 and *__x3 <= *__x4.
  // *__x4 has at least 3 elements <= it, so it cannot be the median.
  if (__comp(*__x4, *__x2)) {
    _Ops::iter_swap(__x1, __x3);
    _Ops::iter_swap(__x2, __x4);
  }

  // Insert __x5 beside __x3.
  if (__comp(*__x5, *__x3))
    _Ops::iter_swap(__x3, __x5);

  // Order pairs (__x1, __x2) and (__x3, __x5) by their larger element:
  // *__x1 <= *__x2 <= *__x5 and *__x3 <= *__x5.
  if (__comp(*__x5, *__x2)) {
    _Ops::iter_swap(__x1, __x3);
    _Ops::iter_swap(__x2, __x5);
  }

  // The median is the larger of __x2 and __x3.
  if (__comp(*__x3, *__x2))
    _Ops::iter_swap(__x2, __x3);
}

// Branchless 7-comparator selection network. One extra comparison vs the adaptive algorithm
// above, but no control flow, which is faster for cheap arithmetic types.
template <class,
          class _Compare,
          class _RandomAccessIterator,
          __enable_if_t<__use_branchless_sort<_Compare, _RandomAccessIterator>, int> = 0>
_LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX14 void
__nth_element_median5(_RandomAccessIterator __first, _Compare __comp) {
  typedef typename iterator_traits<_RandomAccessIterator>::difference_type difference_type;

  _RandomAccessIterator __x1 = __first;
  _RandomAccessIterator __x2 = __first + difference_type(1);
  _RandomAccessIterator __x3 = __first + difference_type(2);
  _RandomAccessIterator __x4 = __first + difference_type(3);
  _RandomAccessIterator __x5 = __first + difference_type(4);

  std::__cond_swap<_Compare>(__x1, __x2, __comp);
  std::__cond_swap<_Compare>(__x3, __x4, __comp);
  std::__cond_swap<_Compare>(__x1, __x3, __comp);
  std::__cond_swap<_Compare>(__x2, __x4, __comp);
  std::__cond_swap<_Compare>(__x2, __x3, __comp);
  std::__cond_swap<_Compare>(__x2, __x5, __comp);
  std::__cond_swap<_Compare>(__x3, __x5, __comp);
}

// nth_element on 4 elements. 3 comparisons for the min or max, 4 for the two inner
// positions (Knuth V_2(4)). A full sort would need 5.
template <class _AlgPolicy,
          class _Compare,
          class _RandomAccessIterator,
          __enable_if_t<!__use_branchless_sort<_Compare, _RandomAccessIterator>, int> = 0>
_LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX14 void
__nth_element4(_RandomAccessIterator __first, _RandomAccessIterator __nth, _Compare __comp) {
  using _Ops = _IterOps<_AlgPolicy>;
  typedef typename iterator_traits<_RandomAccessIterator>::difference_type difference_type;

  _RandomAccessIterator __x1 = __first;
  _RandomAccessIterator __x2 = __first + difference_type(1);
  _RandomAccessIterator __x3 = __first + difference_type(2);
  _RandomAccessIterator __x4 = __first + difference_type(3);

  if (__comp(*__x2, *__x1))
    _Ops::iter_swap(__x1, __x2);
  if (__comp(*__x4, *__x3))
    _Ops::iter_swap(__x3, __x4);

  const difference_type __k = __nth - __first;
  if (__k == 0) {
    if (__comp(*__x3, *__x1))
      _Ops::iter_swap(__x1, __x3);
    return;
  }
  if (__k == 3) {
    if (__comp(*__x4, *__x2))
      _Ops::iter_swap(__x2, __x4);
    return;
  }

  if (__k == 1) {
    // Order pairs by their smaller element so the min is at __x1.
    if (__comp(*__x3, *__x1)) {
      _Ops::iter_swap(__x1, __x3);
      _Ops::iter_swap(__x2, __x4);
    }
  } else {
    // Order pairs by their larger element so the max is at __x4.
    if (__comp(*__x4, *__x2)) {
      _Ops::iter_swap(__x1, __x3);
      _Ops::iter_swap(__x2, __x4);
    }
  }
  // 2nd smallest at __x2 (__k == 1), or 2nd largest at __x3 (__k == 2).
  if (__comp(*__x3, *__x2))
    _Ops::iter_swap(__x2, __x3);
}

template <class _AlgPolicy,
          class _Compare,
          class _RandomAccessIterator,
          __enable_if_t<__use_branchless_sort<_Compare, _RandomAccessIterator>, int> = 0>
_LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX14 void
__nth_element4(_RandomAccessIterator __first, _RandomAccessIterator, _Compare __comp) {
  typedef typename iterator_traits<_RandomAccessIterator>::difference_type difference_type;
  std::__sort4<_AlgPolicy, _Compare>(
      __first, __first + difference_type(1), __first + difference_type(2), __first + difference_type(3), __comp);
}

// nth_element on 3 elements. Min or max takes 2 comparisons; the middle element is a
// median-of-3, which is exactly a 3-sort.
template <class _AlgPolicy,
          class _Compare,
          class _RandomAccessIterator,
          __enable_if_t<!__use_branchless_sort<_Compare, _RandomAccessIterator>, int> = 0>
_LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX14 void
__nth_element3(_RandomAccessIterator __first, _RandomAccessIterator __nth, _Compare __comp) {
  using _Ops = _IterOps<_AlgPolicy>;
  typedef typename iterator_traits<_RandomAccessIterator>::difference_type difference_type;

  _RandomAccessIterator __x1 = __first;
  _RandomAccessIterator __x2 = __first + difference_type(1);
  _RandomAccessIterator __x3 = __first + difference_type(2);

  const difference_type __k = __nth - __first;
  if (__k == 0) {
    if (__comp(*__x2, *__x1))
      _Ops::iter_swap(__x1, __x2);
    if (__comp(*__x3, *__x1))
      _Ops::iter_swap(__x1, __x3);
    return;
  }
  if (__k == 2) {
    if (__comp(*__x2, *__x1))
      _Ops::iter_swap(__x1, __x2);
    if (__comp(*__x3, *__x2))
      _Ops::iter_swap(__x2, __x3);
    return;
  }
  std::__sort3<_AlgPolicy, _Compare>(__x1, __x2, __x3, __comp);
}

template <class _AlgPolicy,
          class _Compare,
          class _RandomAccessIterator,
          __enable_if_t<__use_branchless_sort<_Compare, _RandomAccessIterator>, int> = 0>
_LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX14 void
__nth_element3(_RandomAccessIterator __first, _RandomAccessIterator, _Compare __comp) {
  typedef typename iterator_traits<_RandomAccessIterator>::difference_type difference_type;
  std::__sort3<_AlgPolicy, _Compare>(__first, __first + difference_type(1), __first + difference_type(2), __comp);
}

// Handles ranges of at most 5 elements. Returns false if the range was too large to be handled.
template <class _AlgPolicy, class _Compare, class _RandomAccessIterator>
_LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX14 bool __nth_element_small(
    _RandomAccessIterator __first, _RandomAccessIterator __nth, _RandomAccessIterator __last, _Compare __comp) {
  using _Ops = _IterOps<_AlgPolicy>;
  typedef typename iterator_traits<_RandomAccessIterator>::difference_type difference_type;
  switch (__last - __first) {
  case 0:
  case 1:
    return true;
  case 2:
    if (__comp(*--__last, *__first))
      _Ops::iter_swap(__first, __last);
    return true;
  case 3:
    std::__nth_element3<_AlgPolicy, _Compare>(__first, __nth, __comp);
    return true;
  case 4:
    std::__nth_element4<_AlgPolicy, _Compare>(__first, __nth, __comp);
    return true;
  case 5:
    if (__nth == __first + difference_type(2)) {
      std::__nth_element_median5<_AlgPolicy, _Compare>(__first, __comp);
      return true;
    }
    std::__sort5<_AlgPolicy, _Compare>(
        __first,
        __first + difference_type(1),
        __first + difference_type(2),
        __first + difference_type(3),
        --__last,
        __comp);
    return true;
  }
  return false;
}

template <class _Compare, class _RandomAccessIterator>
_LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX14 bool __nth_element_find_guard(
    _RandomAccessIterator& __i, _RandomAccessIterator& __j, _RandomAccessIterator __m, _Compare __comp) {
  // manually guard downward moving __j against __i
  while (true) {
    if (__i == --__j) {
      return false;
    }
    if (__comp(*__j, *__m)) {
      return true; // found guard for downward moving __j, now use unguarded partition
    }
  }
}

template <class _AlgPolicy, class _Compare, class _RandomAccessIterator>
_LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX14 void
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
__nth_element(
    _RandomAccessIterator __first, _RandomAccessIterator __nth, _RandomAccessIterator __last, _Compare __comp) {
  using _Ops = _IterOps<_AlgPolicy>;

  // _Compare is known to be a reference type
  typedef typename iterator_traits<_RandomAccessIterator>::difference_type difference_type;
  const difference_type __limit = 7;
  while (true) {
    if (__nth == __last)
      return;
    if (std::__nth_element_small<_AlgPolicy, _Compare>(__first, __nth, __last, __comp))
      return;
    difference_type __len = __last - __first;
    if (__len <= __limit) {
      std::__selection_sort<_AlgPolicy, _Compare>(__first, __last, __comp);
      return;
    }
    // __len > __limit >= 3
    _RandomAccessIterator __m   = __first + __len / 2;
    _RandomAccessIterator __lm1 = __last;
    unsigned __n_swaps          = std::__sort3<_AlgPolicy, _Compare>(__first, __m, --__lm1, __comp);
    // *__m is median
    // partition [__first, __m) < *__m and *__m <= [__m, __last)
    // (this inhibits tossing elements equivalent to __m around unnecessarily)
    _RandomAccessIterator __i = __first;
    _RandomAccessIterator __j = __lm1;
    // j points beyond range to be tested, *__lm1 is known to be <= *__m
    // The search going up is known to be guarded but the search coming down isn't.
    // Prime the downward search with a guard.
    if (!__comp(*__i, *__m)) // if *__first == *__m
    {
      // *__first == *__m, *__first doesn't go in first part
      if (std::__nth_element_find_guard<_Compare>(__i, __j, __m, __comp)) {
        _Ops::iter_swap(__i, __j);
        ++__n_swaps;
      } else {
        // *__first == *__m, *__m <= all other elements
        // Partition instead into [__first, __i) == *__first and *__first < [__i, __last)
        ++__i; // __first + 1
        __j = __last;
        if (!__comp(*__first, *--__j)) { // we need a guard if *__first == *(__last-1)
          while (true) {
            if (__i == __j) {
              return; // [__first, __last) all equivalent elements
            } else if (__comp(*__first, *__i)) {
              _Ops::iter_swap(__i, __j);
              ++__n_swaps;
              ++__i;
              break;
            }
            ++__i;
          }
        }
        // [__first, __i) == *__first and *__first < [__j, __last) and __j == __last - 1
        if (__i == __j) {
          return;
        }
        while (true) {
          while (!__comp(*__first, *__i)) {
            ++__i;
            _LIBCPP_ASSERT_VALID_ELEMENT_ACCESS(
                __i != __last,
                "Would read out of bounds, does your comparator satisfy the strict-weak ordering requirement?");
          }
          do {
            _LIBCPP_ASSERT_VALID_ELEMENT_ACCESS(
                __j != __first,
                "Would read out of bounds, does your comparator satisfy the strict-weak ordering requirement?");
            --__j;
          } while (__comp(*__first, *__j));
          if (__i >= __j)
            break;
          _Ops::iter_swap(__i, __j);
          ++__n_swaps;
          ++__i;
        }
        // [__first, __i) == *__first and *__first < [__i, __last)
        // The first part is sorted,
        if (__nth < __i) {
          return;
        }
        // __nth_element the second part
        // std::__nth_element<_Compare>(__i, __nth, __last, __comp);
        __first = __i;
        continue;
      }
    }
    ++__i;
    // j points beyond range to be tested, *__lm1 is known to be <= *__m
    // if not yet partitioned...
    if (__i < __j) {
      // known that *(__i - 1) < *__m
      while (true) {
        // __m still guards upward moving __i
        while (__comp(*__i, *__m)) {
          ++__i;
          _LIBCPP_ASSERT_VALID_ELEMENT_ACCESS(
              __i != __last,
              "Would read out of bounds, does your comparator satisfy the strict-weak ordering requirement?");
        }
        // It is now known that a guard exists for downward moving __j
        do {
          _LIBCPP_ASSERT_VALID_ELEMENT_ACCESS(
              __j != __first,
              "Would read out of bounds, does your comparator satisfy the strict-weak ordering requirement?");
          --__j;
        } while (!__comp(*__j, *__m));
        if (__i >= __j)
          break;
        _Ops::iter_swap(__i, __j);
        ++__n_swaps;
        // It is known that __m != __j
        // If __m just moved, follow it
        if (__m == __i)
          __m = __j;
        ++__i;
      }
    }
    // [__first, __i) < *__m and *__m <= [__i, __last)
    if (__i != __m && __comp(*__m, *__i)) {
      _Ops::iter_swap(__i, __m);
      ++__n_swaps;
    }
    // [__first, __i) < *__i and *__i <= [__i+1, __last)
    if (__nth == __i)
      return;
    if (__n_swaps == 0) {
      // We were given a perfectly partitioned sequence.  Coincidence?
      if (__nth < __i) {
        // Check for [__first, __i) already sorted
        __j = __m = __first;
        while (true) {
          if (++__j == __i) {
            // [__first, __i) sorted
            return;
          }
          if (__comp(*__j, *__m)) {
            // not yet sorted, so sort
            break;
          }
          __m = __j;
        }
      } else {
        // Check for [__i, __last) already sorted
        __j = __m = __i;
        while (true) {
          if (++__j == __last) {
            // [__i, __last) sorted
            return;
          }
          if (__comp(*__j, *__m)) {
            // not yet sorted, so sort
            break;
          }
          __m = __j;
        }
      }
    }
    // __nth_element on range containing __nth
    if (__nth < __i) {
      // std::__nth_element<_Compare>(__first, __nth, __i, __comp);
      __last = __i;
    } else {
      // std::__nth_element<_Compare>(__i+1, __nth, __last, __comp);
      __first = ++__i;
    }
  }
}

template <class _AlgPolicy, class _RandomAccessIterator, class _Compare>
inline _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX20 void __nth_element_impl(
    _RandomAccessIterator __first, _RandomAccessIterator __nth, _RandomAccessIterator __last, _Compare& __comp) {
  if (__nth == __last)
    return;

  std::__debug_randomize_range<_AlgPolicy>(__first, __last);

  // When the size of the range is known at compile time (e.g. in a median-of-5 filter), this
  // dispatch constant-folds to the appropriate small-range algorithm. Otherwise, the check
  // disappears entirely and we fall through to the general algorithm, which handles small ranges
  // itself.
  bool __small_constant_size =
      __builtin_constant_p(__last - __first) &&
      std::__nth_element_small<_AlgPolicy, __comp_ref_type<_Compare> >(__first, __nth, __last, __comp);
  if (!__small_constant_size)
    std::__nth_element<_AlgPolicy, __comp_ref_type<_Compare> >(__first, __nth, __last, __comp);

  std::__debug_randomize_range<_AlgPolicy>(__first, __nth);
  if (__nth != __last) {
    std::__debug_randomize_range<_AlgPolicy>(++__nth, __last);
  }
}

template <class _RandomAccessIterator, class _Compare>
inline _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX20 void
nth_element(_RandomAccessIterator __first, _RandomAccessIterator __nth, _RandomAccessIterator __last, _Compare __comp) {
  std::__nth_element_impl<_ClassicAlgPolicy>(std::move(__first), std::move(__nth), std::move(__last), __comp);
}

template <class _RandomAccessIterator>
inline _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX20 void
nth_element(_RandomAccessIterator __first, _RandomAccessIterator __nth, _RandomAccessIterator __last) {
  std::nth_element(std::move(__first), std::move(__nth), std::move(__last), __less<>());
}

_LIBCPP_END_NAMESPACE_STD

_LIBCPP_POP_MACROS

#endif // _LIBCPP___ALGORITHM_NTH_ELEMENT_H
