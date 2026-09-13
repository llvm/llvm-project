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
#include <__fwd/memory.h>
#include <__memory/allocator_traits.h>
#include <__new/allocate.h>
#include <__type_traits/is_constant_evaluated.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

#if _LIBCPP_STD_VER >= 23

// This function allocates memory using the allocator's allocate_at_least member if possible, and falls back the normal
// allocate in older modes.

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

// Provide an efficient __allocate_at_least for std::allocator in all standard modes

template <class _Tp>
[[__nodiscard__]] _LIBCPP_CONSTEXPR __allocation_result<_Tp*> __allocate_at_least(allocator<_Tp>& __alloc, size_t __n) {
  if (__libcpp_is_constant_evaluated()) {
    return __allocation_result<_Tp*>(__alloc.allocate(__n), __n);
  } else {
    return std::__libcpp_allocate_at_least<_Tp>(__element_count(__n));
  }
}

#endif // _LIBCPP_STD_VER >= 23

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___MEMORY_ALLOCATE_AT_LEAST_H
