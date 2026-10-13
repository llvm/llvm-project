//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___PSTL_STATPART_TRANSFORM_H
#define _LIBCPP___PSTL_STATPART_TRANSFORM_H

#include <__algorithm/transform.h>
#include <__config>
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
struct __transform<__statpart_backend_tag, _ExecutionPolicy> {
  template <class _Policy, class _ForwardIterator, class _ForwardOutIterator, class _UnaryOperation>
  _LIBCPP_HIDE_FROM_ABI optional<_ForwardOutIterator> operator()(
      _Policy&&, _ForwardIterator __first, _ForwardIterator __last, _ForwardOutIterator __result, _UnaryOperation __op)
      const noexcept {
    if constexpr (__is_parallel_execution_policy_v<_ExecutionPolicy> &&
                  __has_random_access_iterator_category_or_concept<_ForwardIterator>::value &&
                  __has_random_access_iterator_category_or_concept<_ForwardOutIterator>::value) {
      __statpart::__for_each(
          __first, __last, [__op, __first, __result](_ForwardIterator __brick_first, _ForwardIterator __brick_last) {
            return std::transform(__brick_first, __brick_last, __result + (__brick_first - __first), __op);
          });
      return __result + (__last - __first);
    } else {
      return std::transform(__first, __last, __result, __op);
    }
  }
};

template <class _ExecutionPolicy>
struct __transform_binary<__statpart_backend_tag, _ExecutionPolicy> {
  template <class _Policy,
            class _ForwardIterator1,
            class _ForwardIterator2,
            class _ForwardOutIterator,
            class _BinaryOperation>
  _LIBCPP_HIDE_FROM_ABI optional<_ForwardOutIterator>
  operator()(_Policy&&,
             _ForwardIterator1 __first1,
             _ForwardIterator1 __last1,
             _ForwardIterator2 __first2,
             _ForwardOutIterator __result,
             _BinaryOperation __op) const noexcept {
    if constexpr (__is_parallel_execution_policy_v<_ExecutionPolicy> &&
                  __has_random_access_iterator_category_or_concept<_ForwardIterator1>::value &&
                  __has_random_access_iterator_category_or_concept<_ForwardIterator2>::value &&
                  __has_random_access_iterator_category_or_concept<_ForwardOutIterator>::value) {
      auto __res = __statpart::__for_each(
          __first1,
          __last1,
          [__op, __first1, __first2, __result](_ForwardIterator1 __brick_first, _ForwardIterator1 __brick_last) {
            return std::transform(
                __brick_first,
                __brick_last,
                __first2 + (__brick_first - __first1),
                __result + (__brick_first - __first1),
                __op);
          });
      if (!__res)
        return nullopt;
      return __result + (__last1 - __first1);
    } else {
      return std::transform(__first1, __last1, __first2, __result, __op);
    }
  }
};

} // namespace __pstl
_LIBCPP_END_EXPLICIT_ABI_ANNOTATIONS
_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP_STD_VER >= 17

_LIBCPP_POP_MACROS

#endif // _LIBCPP___PSTL_STATPART_TRANSFORM_H
