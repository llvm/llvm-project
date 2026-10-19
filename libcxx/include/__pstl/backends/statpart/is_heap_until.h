//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___PSTL_BACKENDS_STATPART_IS_HEAP_UNTIL_H
#define _LIBCPP___PSTL_BACKENDS_STATPART_IS_HEAP_UNTIL_H

#include <__algorithm/is_heap_until.h>
#include <__config>
#include <__execution/is_execution_policy.h>
#include <__functional/operations.h>
#include <__iterator/concepts.h>
#include <__iterator/iterator_traits.h>
#include <__optional/nullopt_t.h>
#include <__optional/optional.h>
#include <__pstl/backend_fwd.h>
#include <__pstl/backends/statpart/basis.h>
#include <__utility/move.h>

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

} // namespace __pstl
_LIBCPP_END_EXPLICIT_ABI_ANNOTATIONS
_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP_STD_VER >= 17

_LIBCPP_POP_MACROS

#endif // _LIBCPP___PSTL_BACKENDS_STATPART_IS_HEAP_UNTIL_H
