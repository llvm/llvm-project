//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___NEW_ALLOCATION_RESULT_H
#define _LIBCPP___NEW_ALLOCATION_RESULT_H

#include <__config>
#include <__cstddef/size_t.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

template <class _Pointer, class _SizeT = size_t>
struct __allocation_result {
  _Pointer ptr;
  _SizeT count;

  _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR __allocation_result(_Pointer __ptr, _SizeT __count)
      : ptr(__ptr), count(__count) {}
};
_LIBCPP_CTAD_SUPPORTED_FOR_TYPE(__allocation_result);

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___NEW_ALLOCATION_RESULT_H
