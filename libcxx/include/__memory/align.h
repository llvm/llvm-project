//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___MEMORY_ALIGN_H
#define _LIBCPP___MEMORY_ALIGN_H

#include <__config>
#include <__cstddef/size_t.h>
#include <cstdint>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

inline namespace __align_inline {
_LIBCPP_HIDE_FROM_ABI inline void* align(size_t __align, size_t __sz, void*& __ptr, size_t& __space) {
  void* __r = nullptr;
  if (__sz <= __space) {
    char* __p1 = static_cast<char*>(__ptr);
    char* __p2 = reinterpret_cast<char*>(reinterpret_cast<uintptr_t>(__p1 + (__align - 1)) & -__align);
    size_t __d = static_cast<size_t>(__p2 - __p1);
    if (__d <= __space - __sz) {
      __r   = __p2;
      __ptr = __r;
      __space -= __d;
    }
  }
  return __r;
}

} // namespace __align_inline

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___MEMORY_ALIGN_H
