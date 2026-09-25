//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14
// TODO: Change to XFAIL once https://llvm.org/PR40995 is fixed
// UNSUPPORTED: availability-pmr-missing

// UNSUPPORTED: sanitizer-new-delete

// XFAIL: using-built-library-before-llvm-24

// <memory_resource>

// memory_resource *new_delete_resource()

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <memory_resource>
#include <new>

alignas(128) char buffer[128];
bool allocated = false;

void* operator new(std::size_t size) {
  assert(!allocated);

  allocated = true;
  if (size <= 4)
    return buffer + 4;
  if (size <= 8)
    return buffer + 8;
  if (size <= 16)
    return buffer + 16;
  if (size <= 128)
    return buffer + 128;

  assert(false && "Unexpected allocation size");
  return buffer;
}

void operator delete(void*) noexcept { allocated = false; }

void* operator new(std::size_t size, std::align_val_t align) {
  assert(static_cast<size_t>(align) <= 128);
  return operator new(std::max(size, static_cast<size_t>(align)));
}

void operator delete(void*, std::align_val_t) noexcept { allocated = false; }

int main(int, char**) {
  std::pmr::memory_resource* res = std::pmr::new_delete_resource();
  void* ptr                      = res->allocate(1, __STDCPP_DEFAULT_NEW_ALIGNMENT__);
  assert((reinterpret_cast<uintptr_t>(ptr) & (__STDCPP_DEFAULT_NEW_ALIGNMENT__ - 1)) == 0);

  return 0;
}
