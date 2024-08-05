// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___MEMORY_TEMPORARY_BUFFER_H
#define _LIBCPP___MEMORY_TEMPORARY_BUFFER_H

#include <__config>
#include <__memory/scoped_temporary_buffer.h>
#include <__utility/pair.h>
#include <cstddef>
#include <new>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

#if _LIBCPP_STD_VER <= 17 || defined(_LIBCPP_ENABLE_CXX20_REMOVED_TEMPORARY_BUFFER)

_LIBCPP_BEGIN_NAMESPACE_STD

template <class _Tp>
_LIBCPP_NODISCARD _LIBCPP_HIDE_FROM_ABI _LIBCPP_NO_CFI _LIBCPP_DEPRECATED_IN_CXX17 pair<_Tp*, ptrdiff_t>
get_temporary_buffer(ptrdiff_t __n) _NOEXCEPT {
  __scoped_temporary_buffer<_Tp> __scoped_buf(__n);
  __temporary_allocation_result<_Tp> __result = __scoped_buf.__release_to_raw();
  return pair<_Tp*, ptrdiff_t>(__result.__ptr, __result.__count);
}

template <class _Tp>
inline _LIBCPP_HIDE_FROM_ABI _LIBCPP_DEPRECATED_IN_CXX17 void return_temporary_buffer(_Tp* __p) _NOEXCEPT {
  __scoped_temporary_buffer<_Tp> __scoped_buf(__p);
  (void)__scoped_buf;
}

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP_STD_VER <= 17 || defined(_LIBCPP_ENABLE_CXX20_REMOVED_TEMPORARY_BUFFER)

#endif // _LIBCPP___MEMORY_TEMPORARY_BUFFER_H
