//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___PSTL_CPU_ALGOS_TRANSFORM_SCAN_H
#define _LIBCPP___PSTL_CPU_ALGOS_TRANSFORM_SCAN_H

#include <__config>
#include <__iterator/concepts.h>
#include <__iterator/iterator_traits.h>
#include <__numeric/transform_exclusive_scan.h>
#include <__numeric/transform_inclusive_scan.h>
#include <__pstl/backend_fwd.h>
#include <__pstl/cpu_algos/cpu_traits.h>
#include <__type_traits/is_execution_policy.h>
#include <__utility/move.h>
#include <optional>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_PUSH_MACROS
#include <__undef_macros>

#if _LIBCPP_STD_VER >= 17

_LIBCPP_BEGIN_NAMESPACE_STD
namespace __pstl {

template <class _Backend, class _RawExecutionPolicy>
struct __cpu_parallel_transform_exclusive_scan {
  template <class _Policy,
            class _ForwardIterator1,
            class _ForwardIterator2,
            class _Tp,
            class _BinaryOperation,
            class _UnaryOperation>
  _LIBCPP_HIDE_FROM_ABI optional<_ForwardIterator2> operator()(
      _Policy&&,
      _ForwardIterator1 __first,
      _ForwardIterator1 __last,
      _ForwardIterator2 __result,
      _Tp __init,
      _BinaryOperation __binary_op,
      _UnaryOperation __unary_op) const noexcept {
    if constexpr (__is_parallel_execution_policy_v<_RawExecutionPolicy> &&
                  __has_random_access_iterator_category_or_concept<_ForwardIterator1>::value &&
                  __has_random_access_iterator_category_or_concept<_ForwardIterator2>::value) {
      return __cpu_traits<_Backend>::__transform_exclusive_scan(
          std::move(__first),
          std::move(__last),
          std::move(__result),
          std::move(__init),
          std::move(__binary_op),
          std::move(__unary_op));
    } else {
      return std::transform_exclusive_scan(
          std::move(__first),
          std::move(__last),
          std::move(__result),
          std::move(__init),
          std::move(__binary_op),
          std::move(__unary_op));
    }
  }
};

template <class _Backend, class _RawExecutionPolicy>
struct __cpu_parallel_transform_inclusive_scan {
  template <class _Policy,
            class _ForwardIterator1,
            class _ForwardIterator2,
            class _BinaryOperation,
            class _UnaryOperation>
  _LIBCPP_HIDE_FROM_ABI optional<_ForwardIterator2>
  operator()(_Policy&&,
             _ForwardIterator1 __first,
             _ForwardIterator1 __last,
             _ForwardIterator2 __result,
             _BinaryOperation __binary_op,
             _UnaryOperation __unary_op) const noexcept {
    if constexpr (__is_parallel_execution_policy_v<_RawExecutionPolicy> &&
                  __has_random_access_iterator_category_or_concept<_ForwardIterator1>::value &&
                  __has_random_access_iterator_category_or_concept<_ForwardIterator2>::value) {
      return __cpu_traits<_Backend>::__transform_inclusive_scan(
          std::move(__first), std::move(__last), std::move(__result), std::move(__binary_op), std::move(__unary_op));
    } else {
      return std::transform_inclusive_scan(
          std::move(__first), std::move(__last), std::move(__result), std::move(__binary_op), std::move(__unary_op));
    }
  }

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
      _BinaryOperation __binary_op,
      _UnaryOperation __unary_op,
      _Tp __init) const noexcept {
    if constexpr (__is_parallel_execution_policy_v<_RawExecutionPolicy> &&
                  __has_random_access_iterator_category_or_concept<_ForwardIterator1>::value &&
                  __has_random_access_iterator_category_or_concept<_ForwardIterator2>::value) {
      return __cpu_traits<_Backend>::__transform_inclusive_scan(
          std::move(__first),
          std::move(__last),
          std::move(__result),
          std::move(__binary_op),
          std::move(__unary_op),
          std::move(__init));
    } else {
      return std::transform_inclusive_scan(
          std::move(__first),
          std::move(__last),
          std::move(__result),
          std::move(__binary_op),
          std::move(__unary_op),
          std::move(__init));
    }
  }
};

} // namespace __pstl
_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP_STD_VER >= 17

_LIBCPP_POP_MACROS

#endif // _LIBCPP___PSTL_CPU_ALGOS_TRANSFORM_SCAN_H
