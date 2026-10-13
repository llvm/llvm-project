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
#include <__pstl/backends/statpart/find_if.h>
#include <__pstl/backends/statpart/for_each.h>
#include <__pstl/backends/statpart/is_heap_until.h>
#include <__pstl/backends/statpart/merge.h>
#include <__pstl/backends/statpart/min_element.h>
#include <__pstl/backends/statpart/minmax_element.h>
#include <__pstl/backends/statpart/mismatch.h>
#include <__pstl/backends/statpart/reverse.h>
#include <__pstl/backends/statpart/search.h>

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
