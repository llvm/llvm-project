//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___PSTL_STATPART_FIND_IF_H
#define _LIBCPP___PSTL_STATPART_FIND_IF_H

#include <__algorithm/find_if.h>
#include <__config>
#include <__functional/operations.h>
#include <__iterator/concepts.h>
#include <__iterator/iterator_traits.h>
#include <__optional/nullopt_t.h>
#include <__optional/optional.h>
#include <__pstl/backend_fwd.h>
#include <__pstl/backends/statpart/basis.h>
#include <__type_traits/is_execution_policy.h>
#include <__utility/empty.h>
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

} // namespace __pstl
_LIBCPP_END_EXPLICIT_ABI_ANNOTATIONS
_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP_STD_VER >= 17

_LIBCPP_POP_MACROS

#endif // _LIBCPP___PSTL_STATPART_FIND_IF_H
