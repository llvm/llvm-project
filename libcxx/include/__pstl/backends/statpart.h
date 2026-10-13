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
