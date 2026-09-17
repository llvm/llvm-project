//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___PSTL_CPU_ALGOS_TRANSFORM_INCLUSIVE_SCAN_INIT_H
#define _LIBCPP___PSTL_CPU_ALGOS_TRANSFORM_INCLUSIVE_SCAN_INIT_H

#include <__config>
#include <__functional/identity.h>
#include <__functional/operations.h>
#include <__iterator/concepts.h>
#include <__iterator/iterator_traits.h>
#include <__numeric/inclusive_scan.h>
#include <__numeric/transform_inclusive_scan.h>
#include <__numeric/transform_reduce.h>
#include <__optional/nullopt_t.h>
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
struct __cpu_parallel_transform_inclusive_scan_init {
  template <class _Policy,
            class _ForwardIterator1,
            class _ForwardIterator2,
            class _BinaryOperation,
            class _UnaryOperation,
            class _Tp>
  _LIBCPP_HIDE_FROM_ABI optional<_ForwardIterator2> operator()(
      _Policy&&,
      _ForwardIterator1 __first,
      _ForwardIterator1 __last,
      _ForwardIterator2 __result,
      _BinaryOperation __reduce,
      _UnaryOperation __transform,
      _Tp __init) const noexcept {
    if constexpr (__is_parallel_execution_policy_v<_RawExecutionPolicy> &&
                  __has_random_access_iterator_category_or_concept<_ForwardIterator1>::value &&
                  __has_random_access_iterator_category_or_concept<_ForwardIterator2>::value) {
      auto __ret = __cpu_traits<_Backend>::__scan(
          __first,
          __last,
          __result,
          std::move(__init),
          [&](auto __chunk_first, auto __chunk_last) {
            return std::transform_reduce(
                __chunk_first + 1, __chunk_last, __transform(*__chunk_first), __reduce, __transform);
          },
          [&](auto __prefix_first, auto __prefix_last, auto __init_val) {
            std::inclusive_scan(__prefix_first, __prefix_last, __prefix_first, __reduce, std::move(__init_val));
          },
          [&](auto __chunk_first, auto __chunk_last, auto __chunk_result, auto __chunk_init) {
            std::transform_inclusive_scan(
                __chunk_first, __chunk_last, __chunk_result, __reduce, __transform, std::move(__chunk_init));
          });
      if (!__ret)
        return nullopt;
      return __result + (__last - __first);
    } else {
      // Non-random access iterators cannot be processed in parallel, fall back to the sequential implementation.
      return std::transform_inclusive_scan(
          std::move(__first),
          std::move(__last),
          std::move(__result),
          std::move(__reduce),
          std::move(__transform),
          std::move(__init));
    }
  }
};

} // namespace __pstl
_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP_STD_VER >= 17

_LIBCPP_POP_MACROS

#endif // _LIBCPP___PSTL_CPU_ALGOS_TRANSFORM_INCLUSIVE_SCAN_INIT_H
