//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26
// UNSUPPORTED: libcpp-has-no-experimental-hive

// <hive>

// ~hive();

#include <cassert>
#include <hive>
#include <memory>
#include <type_traits>

#include "Counter.h"
#include "min_allocator.h"
#include "test_allocator.h"
#include "test_macros.h"

static_assert(std::is_nothrow_destructible_v<std::hive<int>>);
static_assert(std::is_nothrow_destructible_v<std::hive<int, min_allocator<int>>>);

// Every live element is destroyed.
template <class A>
void test_destroys_elements() {
  {
    std::hive<Counter<int>, A> h(std::hive_limits(3, 3));
    for (int i = 0; i != 7; ++i) {
      h.insert(Counter<int>(i));
    }
    assert(Counter_base::gConstructed == 7);
  }
  assert(Counter_base::gConstructed == 0);
}

// Every block is freed, active and reserved, and the allocator destroys each element.
void test_frees_every_block() {
  test_allocator_statistics stats;
  {
    std::hive<int, test_allocator<int>> h(std::hive_limits(3, 3), test_allocator<int>(&stats));
    for (int i = 0; i != 7; ++i) {
      h.insert(i);
    }
    assert(stats.alloc_count > 0);
    h.clear();
    LIBCPP_ASSERT(stats.alloc_count > 0);
    h.insert(1);
  }
  assert(stats.alloc_count == 0);
  assert(stats.destroy_count == 8);
}

int main(int, char**) {
  test_destroys_elements<std::allocator<Counter<int>>>();
  test_destroys_elements<min_allocator<Counter<int>>>();
  test_frees_every_block();

  return 0;
}
