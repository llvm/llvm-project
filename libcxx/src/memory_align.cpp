//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <__config>
#include <__cstddef/size_t.h>
#include <cstdint>

// Don't include <memory> to avoid multiple declarations of std::align()

#if !defined(_LIBCPP_ABI_DO_NOT_EXPORT_ALIGN)

_LIBCPP_BEGIN_NAMESPACE_STD

_LIBCPP_EXPORTED_FROM_ABI void* align(size_t alignment, size_t size, void*& ptr, size_t& space) {
  void* r = nullptr;
  if (size <= space) {
    char* p1 = static_cast<char*>(ptr);
    char* p2 = reinterpret_cast<char*>(reinterpret_cast<uintptr_t>(p1 + (alignment - 1)) & -alignment);
    size_t d = static_cast<size_t>(p2 - p1);
    if (d <= space - size) {
      r   = p2;
      ptr = r;
      space -= d;
    }
  }
  return r;
}

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP_ABI_DO_NOT_EXPORT_ALIGN
