//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14
// XFAIL: availability-pmr-missing

// test_memory_resource requires RTTI for dynamic_cast
// UNSUPPORTED: no-rtti

// <memory_resource>

// template <class T> class polymorphic_allocator

// polymorphic_allocator<T>::polymorphic_allocator(memory_resource *)

#include <memory_resource>
#include <type_traits>
#include <cassert>

#include "test_std_memory_resource.h"

int main(int, char**) {
  {
    typedef std::pmr::polymorphic_allocator<void> A;
    static_assert(std::is_convertible_v<decltype(nullptr), A>);
    static_assert(std::is_convertible_v<std::pmr::memory_resource*, A>);
  }
  {
    typedef std::pmr::polymorphic_allocator<void> A;
    TestResource R;
    A const a(&R);
    assert(a.resource() == &R);
  }

  return 0;
}
