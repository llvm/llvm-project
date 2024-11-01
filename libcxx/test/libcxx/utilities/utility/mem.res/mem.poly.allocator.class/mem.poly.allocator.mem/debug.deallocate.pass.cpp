//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: !libcpp-has-legacy-debug-mode, c++03, c++11, c++14
// REQUIRES: has-unix-headers

// <memory_resource>

// template <class T> class polymorphic_allocator

// T* polymorphic_allocator<T>::deallocate(T*, size_t size)

int AssertCount = 0;

#include <memory_resource>
#include <type_traits>
#include <cassert>

#include "check_assertion.h"
#include "test_std_memory_resource.h"

int main(int, char**) {
  using Alloc  = std::pmr::polymorphic_allocator<int>;
  using Traits = std::allocator_traits<Alloc>;
  NullResource R;
  Alloc a(&R);
  const std::size_t maxSize = Traits::max_size(a);

  a.deallocate(nullptr, maxSize); // no assertion
  TEST_LIBCPP_ASSERT_FAILURE(a.deallocate(nullptr, maxSize + 1), "deallocate called for size which exceeds max_size()");

  return 0;
}
