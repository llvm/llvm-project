//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___PSTL_BACKENDS_STATPART_MERGE_H
#define _LIBCPP___PSTL_BACKENDS_STATPART_MERGE_H

#include <__algorithm/merge.h>
#include <__config>
#include <__execution/is_execution_policy.h>
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
struct __merge<__statpart_backend_tag, _ExecutionPolicy> {
  template <class _Policy, class _ForwardIterator1, class _ForwardIterator2, class _ForwardOutIterator, class _Comp>
  _LIBCPP_HIDE_FROM_ABI optional<_ForwardOutIterator> operator()(
      _Policy&&,
      _ForwardIterator1 __first1,
      _ForwardIterator1 __last1,
      _ForwardIterator2 __first2,
      _ForwardIterator2 __last2,
      _ForwardOutIterator __result,
      _Comp __comp) const noexcept {
    if constexpr (__is_parallel_execution_policy_v<_ExecutionPolicy> &&
                  __has_random_access_iterator_category_or_concept<_ForwardIterator1>::value &&
                  __has_random_access_iterator_category_or_concept<_ForwardIterator2>::value &&
                  __has_random_access_iterator_category_or_concept<_ForwardOutIterator>::value) {
      auto __res = __statpart::__merge(
          __first1,
          __last1,
          __first2,
          __last2,
          __result,
          __comp,
          [](_ForwardIterator1 __brick_first1,
             _ForwardIterator1 __brick_last1,
             _ForwardIterator2 __brick_first2,
             _ForwardIterator2 __brick_last2,
             _ForwardOutIterator __brick_result,
             _Comp __g_comp) {
            std::merge(std::move(__brick_first1),
                       std::move(__brick_last1),
                       std::move(__brick_first2),
                       std::move(__brick_last2),
                       std::move(__brick_result),
                       std::move(__g_comp));
          });
      if (!__res)
        return nullopt;
      return __result + (__last1 - __first1) + (__last2 - __first2);
    } else {
      return std::merge(__first1, __last1, __first2, __last2, __result, __comp);
    }
  }
};

} // namespace __pstl
_LIBCPP_END_EXPLICIT_ABI_ANNOTATIONS
_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP_STD_VER >= 17

_LIBCPP_POP_MACROS

#endif // _LIBCPP___PSTL_BACKENDS_STATPART_MERGE_H
