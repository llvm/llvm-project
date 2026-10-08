//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___PSTL_CPU_ALGOS_MISMATCH_H
#define _LIBCPP___PSTL_CPU_ALGOS_MISMATCH_H

#include <__algorithm/min.h>
#include <__algorithm/mismatch.h>
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
struct __cpu_parallel_mismatch {
  template <class _Policy, class _ForwardIterator1, class _ForwardIterator2, class _Predicate>
  _LIBCPP_HIDE_FROM_ABI optional<pair<_ForwardIterator1, _ForwardIterator2>>
  operator()(_Policy&&,
             _ForwardIterator1 __first1,
             _ForwardIterator1 __last1,
             _ForwardIterator2 __first2,
             _ForwardIterator2 __last2,
             _Predicate __pred) const noexcept {
    if constexpr (__is_parallel_execution_policy_v<_RawExecutionPolicy> &&
                  __has_random_access_iterator_category_or_concept<_ForwardIterator1>::value &&
                  __has_random_access_iterator_category_or_concept<_ForwardIterator2>::value) {
      // Look for a mismatch only in the prefix of the two ranges.
      auto __n = std::min(__last1 - __first1, __last2 - __first2);
      // Find a position in the first range where the predicate is false against the corresponding position in the
      // second range.
      auto __res = __pstl::__parallel_find<_Backend>(
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

} // namespace __pstl
_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP_STD_VER >= 17

_LIBCPP_POP_MACROS

#endif // _LIBCPP___PSTL_CPU_ALGOS_MISMATCH_H
