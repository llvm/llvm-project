//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___PSTL_CPU_ALGOS_MINMAX_ELEMENT_H
#define _LIBCPP___PSTL_CPU_ALGOS_MINMAX_ELEMENT_H

#include <__algorithm/minmax_element.h>
#include <__config>
#include <__functional/identity.h>
#include <__functional/operations.h>
#include <__iterator/concepts.h>
#include <__iterator/iterator_traits.h>
#include <__optional/optional.h>
#include <__pstl/backend_fwd.h>
#include <__pstl/cpu_algos/cpu_traits.h>
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
struct __cpu_parallel_minmax_element {
  template <class _Policy, class _ForwardIterator, class _Compare>
  _LIBCPP_HIDE_FROM_ABI optional<pair<_ForwardIterator, _ForwardIterator>>
  operator()(_Policy&&, _ForwardIterator __first, _ForwardIterator __last, _Compare __comp) const noexcept {
    if constexpr (__is_parallel_execution_policy_v<_RawExecutionPolicy> &&
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
      return __cpu_traits<_Backend>::__transform_reduce(
          std::move(__first),
          std::move(__last),
          [](auto __it) { return _IterPair{__it, __it}; }, // Transform an iterator into an iterator pair
          std::move(__init),                               // Use the pair of first iterators as the init element
          __iter_reduce,                                   // Reduction of 2 minmax pairs
          [&__iter_reduce, &__comp](auto __brick_first, auto __brick_last, auto __brick_init) {
            // Reduction of an iterator range + init element: use the serial version to find the minmax among
            // the iterators and then reduce it with the init element.
            // Atm __transform_reduce can give empty bricks in edge cases, handle them explicitly until the contract is
            // tightened.
            return __brick_first == __brick_last
                     ? __brick_init
                     : __iter_reduce(__brick_init, std::minmax_element(__brick_first, __brick_last, __comp));
          });
    } else {
      // Non-random access iterators cannot be processed in parallel, fall back to the sequential implementation.
      return std::minmax_element(std::move(__first), std::move(__last), std::move(__comp));
    }
  }
};

} // namespace __pstl
_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP_STD_VER >= 17

_LIBCPP_POP_MACROS

#endif // _LIBCPP___PSTL_CPU_ALGOS_MINMAX_ELEMENT_H
