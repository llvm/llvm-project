//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___CXX03___MEMORY_ALIGNED_ALLOC_H
#define _LIBCPP___CXX03___MEMORY_ALIGNED_ALLOC_H

#include <__cxx03/__config>
#include <__cxx03/cstddef>
#include <__cxx03/cstdlib>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

#ifndef _LIBCPP_HAS_NO_LIBRARY_ALIGNED_ALLOCATION

// Low-level helpers to call the aligned allocation and deallocation functions
// on the target platform. This is used to implement libc++'s own memory
// allocation routines -- if you need to allocate memory inside the library,
// chances are that you want to use `__libcpp_allocate` instead.
//
// Returns the allocated memory, or `nullptr` on failure.
inline _LIBCPP_HIDE_FROM_ABI void* __libcpp_aligned_alloc(std::size_t __alignment, std::size_t __size) {
#  if defined(_LIBCPP_MSVCRT_LIKE)
  return ::_aligned_malloc(__size, __alignment);
#  else
  void* __result = nullptr;
  (void)::posix_memalign(&__result, __alignment, __size);
  // If posix_memalign fails, __result is unmodified so we still return `nullptr`.
  return __result;
#  endif
}

inline _LIBCPP_HIDE_FROM_ABI void __libcpp_aligned_free(void* __ptr) {
#  if defined(_LIBCPP_MSVCRT_LIKE)
  ::_aligned_free(__ptr);
#  else
  ::free(__ptr);
#  endif
}

#endif // !_LIBCPP_HAS_NO_LIBRARY_ALIGNED_ALLOCATION

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___CXX03___MEMORY_ALIGNED_ALLOC_H
