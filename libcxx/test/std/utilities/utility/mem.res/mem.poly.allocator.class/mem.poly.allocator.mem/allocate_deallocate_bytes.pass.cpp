//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// XFAIL: use_system_cxx_lib && target={{.+}}-apple-macosx10.{{9|10|11|12|13|14|15}}
// XFAIL: use_system_cxx_lib && target={{.+}}-apple-macosx{{11.0|12.0}}

// test_memory_resource requires RTTI for dynamic_cast
// UNSUPPORTED: no-rtti

// <memory_resource>

// polymorphic_allocator::allocate_bytes()
// polymorphic_allocator::deallocate_bytes()

#include <algorithm>
#include <cassert>
#include <concepts>
#include <memory_resource>

#include "tracking_mem_res.h"

template <class T>
void test() {
  size_t last_size      = 0;
  size_t last_alignment = 0;
  TrackingMemRes resource(&last_size, &last_alignment);

  std::pmr::polymorphic_allocator<T> allocator(&resource);

  {
    std::same_as<void*> decltype(auto) allocation = allocator.allocate_bytes(13);
    auto ptr                                      = static_cast<char*>(allocation);
    std::fill(ptr, ptr + 13, '0');
    assert(last_size == 13);
    assert(last_alignment == alignof(max_align_t));
    allocator.deallocate_bytes(allocation, 13);
    assert(last_size == 13);
    assert(last_alignment == alignof(max_align_t));
  }
  {
    void* allocation = allocator.allocate_bytes(13, 64);
    auto ptr         = static_cast<char*>(allocation);
    std::fill(ptr, ptr + 13, '0');
    assert(last_size == 13);
    assert(last_alignment == 64);
    allocator.deallocate_bytes(allocation, 13, 64);
    assert(last_size == 13);
    assert(last_alignment == 64);
  }
}

struct S {};

int main(int, char**) {
  test<std::byte>();
  test<S>();

  return 0;
}
