// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___PSTL_BACKENDS_SERIAL_H
#define _LIBCPP___PSTL_BACKENDS_SERIAL_H

#include <__config>
#include <__pstl/configuration_fwd.h>
#include <__pstl/cpu_algos/cpu_traits.h>
#include <__utility/empty.h>
#include <__utility/move.h>
#include <cstddef>
#include <optional>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

#if !defined(_LIBCPP_HAS_NO_INCOMPLETE_PSTL) && _LIBCPP_STD_VER >= 17

_LIBCPP_PUSH_MACROS
#  include <__undef_macros>

_LIBCPP_BEGIN_NAMESPACE_STD
namespace __pstl {

template <>
struct __cpu_traits<__serial_backend_tag> {
  template <class _RandomAccessIterator, class _Fp>
  _LIBCPP_HIDE_FROM_ABI static optional<__empty>
  __for_each(_RandomAccessIterator __first, _RandomAccessIterator __last, _Fp __f) {
    __f(__first, __last);
    return __empty{};
  }

  template <class _Index, class _UnaryOp, class _Tp, class _BinaryOp, class _Reduce>
  _LIBCPP_HIDE_FROM_ABI static optional<_Tp>
  __transform_reduce(_Index __first, _Index __last, _UnaryOp, _Tp __init, _BinaryOp, _Reduce __reduce) {
    return __reduce(std::move(__first), std::move(__last), std::move(__init));
  }

  template <class _RandomAccessIterator, class _Compare, class _LeafSort>
  _LIBCPP_HIDE_FROM_ABI static optional<__empty>
  __stable_sort(_RandomAccessIterator __first, _RandomAccessIterator __last, _Compare __comp, _LeafSort __leaf_sort) {
    __leaf_sort(__first, __last, __comp);
    return __empty{};
  }

  _LIBCPP_HIDE_FROM_ABI static void __cancel_execution() {}

  template <class _RandomAccessIterator1,
            class _RandomAccessIterator2,
            class _RandomAccessIterator3,
            class _Compare,
            class _LeafMerge>
  _LIBCPP_HIDE_FROM_ABI static optional<__empty>
  __merge(_RandomAccessIterator1 __first1,
          _RandomAccessIterator1 __last1,
          _RandomAccessIterator2 __first2,
          _RandomAccessIterator2 __last2,
          _RandomAccessIterator3 __outit,
          _Compare __comp,
          _LeafMerge __leaf_merge) {
    __leaf_merge(__first1, __last1, __first2, __last2, __outit, __comp);
    return __empty{};
  }

  static constexpr size_t __lane_size = 64;
};

} // namespace __pstl
_LIBCPP_END_NAMESPACE_STD

_LIBCPP_POP_MACROS

#endif // !defined(_LIBCPP_HAS_NO_INCOMPLETE_PSTL) && && _LIBCPP_STD_VER >= 17

// Implement PSTL algorithms based on the __cpu_traits specialized above
#include <__pstl/cpu_algos/any_of.h>
#include <__pstl/cpu_algos/fill.h>
#include <__pstl/cpu_algos/find_if.h>
#include <__pstl/cpu_algos/for_each.h>
#include <__pstl/cpu_algos/merge.h>
#include <__pstl/cpu_algos/stable_sort.h>
#include <__pstl/cpu_algos/transform.h>
#include <__pstl/cpu_algos/transform_reduce.h>

#endif // _LIBCPP___PSTL_BACKENDS_SERIAL_H
