//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___ALGORITHM_REMOVE_H
#define _LIBCPP___ALGORITHM_REMOVE_H

#include <__algorithm/find.h>
#include <__algorithm/find_if.h>
#include <__algorithm/simd_utils.h>
#include <__algorithm/unwrap_iter.h>
#include <__bit/popcount.h>
#include <__config>
#include <__utility/move.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_PUSH_MACROS
#include <__undef_macros>

_LIBCPP_BEGIN_NAMESPACE_STD

template <class _Tp, __enable_if_t<__has_compressstore<__simd_vector<_Tp, __native_vector_size<_Tp>>>, int> = 0>
_LIBCPP_NODISCARD _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX20 _Tp*
__remove(_Tp* __first, _Tp* __last, const _Tp& __val) {
  __first = std::find(__first, __last, __val);
  constexpr size_t __vec_size = __native_vector_size<_Tp>;
  using __vec                 = __simd_vector<_Tp, __vec_size>;

  auto __vals = std::__broadcast<__vec>(__val);
  _Tp* __out  = __first;

  while (static_cast<size_t>(__last - __first) >= __vec_size) {
    auto __elements = std::__load_vector<__vec>(__first);
    auto __cmp      = __elements != __vals;
    std::__compressstore(__out, __elements, __cmp);
    __out += std::__popcount(std::__to_int_mask(__cmp));
    __first += __vec_size;
  }
  for (; __first != __last; ++__first) {
    if (*__first != __val)
      *__out++ = *__first;
  }
  return __out;
}

template <class _Iter, class _Tp>
_LIBCPP_NODISCARD _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX20 _Iter
__remove(_Iter __first, _Iter __last, const _Tp& __value) {
  __first = std::find(__first, __last, __value);
  if (__first != __last) {
    _Iter __i = __first;
    while (++__i != __last) {
      if (!(*__i == __value)) {
        *__first = std::move(*__i);
        ++__first;
      }
    }
  }
  return __first;
}

template <class _ForwardIterator, class _Tp>
_LIBCPP_NODISCARD _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX20 _ForwardIterator
remove(_ForwardIterator __first, _ForwardIterator __last, const _Tp& __value) {
  return std::__rewrap_iter(__first, std::__remove(std::__unwrap_iter(__first), std::__unwrap_iter(__last), __value));
}

_LIBCPP_END_NAMESPACE_STD

_LIBCPP_POP_MACROS

#endif // _LIBCPP___ALGORITHM_REMOVE_H
