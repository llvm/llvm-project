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

// polymorphic_allocator::new_object()
// polymorphic_allocator::delete_object()

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
    std::same_as<int*> decltype(auto) allocation = allocator.template new_object<int>();
    std::fill(allocation, allocation + 1, 4);
    assert(last_size == sizeof(int));
    assert(last_alignment == alignof(int));
    allocator.delete_object(allocation);
  }
  {
    std::same_as<int*> decltype(auto) allocation = allocator.template new_object<int>(3);
    assert(*allocation == 3);
    std::fill(allocation, allocation + 1, 4);
    assert(last_size == sizeof(int));
    assert(last_alignment == alignof(int));
    allocator.delete_object(allocation);
  }
  {
    struct TrackConstruction {
      bool* is_constructed_;
      TrackConstruction(bool* is_constructed) : is_constructed_(is_constructed) { *is_constructed = true; }
      ~TrackConstruction() { *is_constructed_ = false; }
    };

    bool is_constructed = false;

    std::same_as<TrackConstruction*> decltype(auto) allocation =
        allocator.template new_object<TrackConstruction>(&is_constructed);
    assert(is_constructed);
    assert(last_size == sizeof(TrackConstruction));
    assert(last_alignment == alignof(TrackConstruction));
    allocator.delete_object(allocation);
    assert(!is_constructed);
  }
}

struct S {};

int main(int, char**) {
  test<std::byte>();
  test<S>();

  return 0;
}
