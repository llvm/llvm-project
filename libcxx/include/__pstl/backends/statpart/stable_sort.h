//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___PSTL_BACKENDS_STATPART_STABLE_SORT_H
#define _LIBCPP___PSTL_BACKENDS_STATPART_STABLE_SORT_H

#include <__algorithm/stable_sort.h>
#include <__config>
#include <__execution/is_execution_policy.h>
#include <__iterator/concepts.h>
#include <__iterator/iterator_traits.h>
#include <__optional/nullopt_t.h>
#include <__optional/optional.h>
#include <__pstl/backend_fwd.h>
#include <__pstl/backends/statpart/basis.h>
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

} // namespace __pstl
_LIBCPP_END_EXPLICIT_ABI_ANNOTATIONS
_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP_STD_VER >= 17

_LIBCPP_POP_MACROS

#endif // _LIBCPP___PSTL_BACKENDS_STATPART_STABLE_SORT_H
