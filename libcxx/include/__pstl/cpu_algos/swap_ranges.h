//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___PSTL_CPU_ALGOS_SWAP_RANGES_H
#define _LIBCPP___PSTL_CPU_ALGOS_SWAP_RANGES_H

#include <__algorithm/swap_ranges.h>
#include <__config>
#include <__iterator/concepts.h>
#include <__iterator/iterator_traits.h>
#include <__optional/nullopt_t.h>
#include <__optional/optional.h>
#include <__pstl/backend_fwd.h>
#include <__pstl/cpu_algos/cpu_traits.h>
#include <__pstl/cpu_algos/for_each.h>
#include <__type_traits/is_execution_policy.h>
#include <__utility/move.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_PUSH_MACROS
#include <__undef_macros>

#if _LIBCPP_STD_VER >= 17

_LIBCPP_BEGIN_NAMESPACE_STD

namespace __pstl {

template <class _Backend, class _RawExecutionPolicy>
struct __cpu_parallel_swap_ranges {
  template <class _Policy, class _ForwardIterator1, class _ForwardIterator2>
  _LIBCPP_HIDE_FROM_ABI optional<_ForwardIterator2> operator()(
      _Policy&&, _ForwardIterator1 __first1, _ForwardIterator1 __last1, _ForwardIterator2 __first2) const noexcept {
    if constexpr (__is_parallel_execution_policy_v<_RawExecutionPolicy> &&
                  __has_random_access_iterator_category_or_concept<_ForwardIterator1>::value &&
                  __has_random_access_iterator_category_or_concept<_ForwardIterator2>::value) {
      auto __res = __pstl::__parallel_for_each_iter_pair<_Backend>(
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

} // namespace __pstl

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP_STD_VER >= 17

_LIBCPP_POP_MACROS

#endif // _LIBCPP___PSTL_CPU_ALGOS_SWAP_RANGES_H
