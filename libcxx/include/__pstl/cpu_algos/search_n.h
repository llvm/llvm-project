//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___PSTL_CPU_ALGOS_SEARCH_N_H
#define _LIBCPP___PSTL_CPU_ALGOS_SEARCH_N_H

#include <__algorithm/search_n.h>
#include <__config>
#include <__functional/operations.h>
#include <__iterator/concepts.h>
#include <__iterator/iterator_traits.h>
#include <__optional/nullopt_t.h>
#include <__optional/optional.h>
#include <__pstl/backend_fwd.h>
#include <__pstl/cpu_algos/cpu_traits.h>
#include <__pstl/cpu_algos/find_if.h>
#include <__type_traits/is_execution_policy.h>
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

template <class _Backend, class _RawExecutionPolicy>
struct __cpu_parallel_search_n {
  template <class _Policy, class _ForwardIterator, class _Size, class _Tp, class _Predicate>
  _LIBCPP_HIDE_FROM_ABI optional<_ForwardIterator>
  operator()(_Policy&&,
             _ForwardIterator __first,
             _ForwardIterator __last,
             _Size __count,
             const _Tp& __value,
             _Predicate __pred) const noexcept {
    if constexpr (__is_parallel_execution_policy_v<_RawExecutionPolicy> &&
                  __has_random_access_iterator_category_or_concept<_ForwardIterator>::value) {
      typedef typename std::iterator_traits<_ForwardIterator>::difference_type _DifferenceType;
      if (static_cast<_DifferenceType>(__count) <= 0) {
        return __first; // If the count is non-positive, the first iterator is returned.
      }
      _DifferenceType __size = __last - __first;
      if (__size < static_cast<_DifferenceType>(__count)) {
        return __last; // The range is too small to contain the requested number of consecutive elements.
      }
      // We're only interested in the range where a potential match can start.
      _DifferenceType __n      = __size - static_cast<_DifferenceType>(__count) + 1;
      _ForwardIterator __last2 = __first + __n;
      // Run a parallel chunked find_if, covering the range where a potential match can start.
      auto __res = __pstl::__parallel_find<_Backend>(
          __first,
          __last2,
          [__count, &__value, &__pred](_ForwardIterator __brick_first, _ForwardIterator __brick_last) {
            // [__brick_first, __brick_first + __count) is guaranteed to be a valid range.
            // __brick_first can walk past __brick_last.
            for (; __brick_first < __brick_last; ++__brick_first) {
              if (!__pred(*__brick_first, __value)) {
                continue;
              }
              // We found a start of a potential match, now check the next __count - 1 elements.
              _ForwardIterator __match_start = __brick_first;
              _Size __matches                = static_cast<_Size>(0);
              while (true) {
                if (++__matches == __count) {
                  return __match_start; // Found a full match, return the start of the match.
                }
                ++__brick_first;
                if (!__pred(*__brick_first, __value)) {
                  break; // The match was broken, continue searching for a new potential match.
                }
              }
            }
            return __brick_last; // No matches starting in this chunk, return the end of the chunk.
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

} // namespace __pstl
_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP_STD_VER >= 17

_LIBCPP_POP_MACROS

#endif // _LIBCPP___PSTL_CPU_ALGOS_SEARCH_N_H
