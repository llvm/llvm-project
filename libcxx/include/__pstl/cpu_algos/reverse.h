//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___PSTL_CPU_ALGOS_REVERSE_H
#define _LIBCPP___PSTL_CPU_ALGOS_REVERSE_H

#include <__algorithm/reverse.h>
#include <__algorithm/swap_ranges.h>
#include <__config>
#include <__iterator/concepts.h>
#include <__iterator/iterator_traits.h>
#include <__iterator/reverse_iterator.h>
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
struct __cpu_parallel_reverse {
  template <class _Policy, class _ForwardIterator>
  _LIBCPP_HIDE_FROM_ABI optional<__empty>
  operator()(_Policy&&, _ForwardIterator __first, _ForwardIterator __last) const noexcept {
    if constexpr (__is_parallel_execution_policy_v<_RawExecutionPolicy> &&
                  __has_random_access_iterator_category_or_concept<_ForwardIterator>::value) {
      // Perform a chunked for_each on the first half of the range.
      auto __n = (__last - __first) / 2;
      return __cpu_traits<_Backend>::__for_each(
          __first, __first + __n, [__first, __last](_ForwardIterator __i, _ForwardIterator __j) {
            // Derive the last position of the mirrored range.
            _ForwardIterator __mirror_last = __last - (__i - __first);
            // Swap the elements in the range of the first half with their mirrored counterparts in the second half.
            std::swap_ranges(
                std::move(__i), std::move(__j), std::reverse_iterator<_ForwardIterator>(std::move(__mirror_last)));
          });
    } else {
      // Non-random access iterators currently cannot be processed in parallel, use the sequential implementation.
      std::reverse(std::move(__first), std::move(__last));
      return __empty{};
    }
  }
};

} // namespace __pstl
_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP_STD_VER >= 17

_LIBCPP_POP_MACROS

#endif // _LIBCPP___PSTL_CPU_ALGOS_REVERSE_H
