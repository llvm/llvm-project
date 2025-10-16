//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___MEMORY_ALLOCATE_AT_LEAST_H
#define _LIBCPP___MEMORY_ALLOCATE_AT_LEAST_H

#include <__config>
#include <__cstddef/size_t.h>
#include <__memory/allocator_traits.h>

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

#if _LIBCPP_STD_VER >= 23

template <class _Alloc>
[[nodiscard]] _LIBCPP_HIDE_FROM_ABI constexpr auto __allocate_at_least(_Alloc& __alloc, size_t __n) {
  auto __res = std::allocator_traits<_Alloc>::allocate_at_least(__alloc, __n);
  return __allocation_result{__res.ptr, __res.count};
}

#else

template <class _Alloc, class _Traits = allocator_traits<_Alloc> >
[[__nodiscard__]] _LIBCPP_HIDE_FROM_ABI
_LIBCPP_CONSTEXPR __allocation_result<typename _Traits::pointer, typename _Traits::size_type>
__allocate_at_least(_Alloc& __alloc, size_t __n) {
  return __allocation_result<typename _Traits::pointer, typename _Traits::size_type>(__alloc.allocate(__n), __n);
}

#endif // _LIBCPP_STD_VER >= 23

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___MEMORY_ALLOCATE_AT_LEAST_H
