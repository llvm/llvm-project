//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___PSTL_CPU_ALGOS_MIN_ELEMENT_H
#define _LIBCPP___PSTL_CPU_ALGOS_MIN_ELEMENT_H

#include <__algorithm/min_element.h>
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

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_PUSH_MACROS
#include <__undef_macros>

#if _LIBCPP_STD_VER >= 17

_LIBCPP_BEGIN_NAMESPACE_STD
namespace __pstl {

template <class _Backend, class _RawExecutionPolicy>
struct __cpu_parallel_min_element {
  template <class _Policy, class _ForwardIterator, class _Compare>
  _LIBCPP_HIDE_FROM_ABI optional<_ForwardIterator>
  operator()(_Policy&&, _ForwardIterator __first, _ForwardIterator __last, _Compare __comp) const noexcept {
    if constexpr (__is_parallel_execution_policy_v<_RawExecutionPolicy> &&
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
      return __cpu_traits<_Backend>::__transform_reduce(
          std::move(__first),
          std::move(__last),
          __identity{},      // No transformations
          std::move(__init), // Use the first iterator as the init element
          __iter_reduce,     // Reduction of 2 elements
          [&__iter_reduce, &__comp](auto __brick_first, auto __brick_last, auto __brick_init) {
            // Reduction of an iterage range + init element: use the serial version to find the minimum among
            // the iterators and then reduce with the init element.
            // Atm __transform_reduce can give empty bricks in edge cases, handle them explicitly until the contract is
            // tightened.
            return __brick_first == __brick_last
                     ? __brick_init
                     : __iter_reduce(__brick_init, std::min_element(__brick_first, __brick_last, __comp));
          });
    } else {
      // Non-random access iterators cannot be processed in parallel, fall back to the sequential implementation.
      return std::min_element(std::move(__first), std::move(__last), std::move(__comp));
    }
  }
};

} // namespace __pstl
_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP_STD_VER >= 17

_LIBCPP_POP_MACROS

#endif // _LIBCPP___PSTL_CPU_ALGOS_MIN_ELEMENT_H
