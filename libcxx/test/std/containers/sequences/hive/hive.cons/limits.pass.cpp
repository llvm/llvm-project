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

// constexpr explicit hive(hive_limits block_limits);
// constexpr hive(hive_limits block_limits, const Allocator&);

#include <cassert>
#include <hive>
#include <initializer_list>
#include <memory>
#include <type_traits>

#include "../helpers.h"
#include "min_allocator.h"
#include "test_allocator.h"

static_assert(!std::is_convertible_v<std::hive_limits, std::hive<int>>);

constexpr std::hive_limits constant_limits = std::hive<int>::block_capacity_default_limits();
constinit std::hive<int> constant_with_limits(constant_limits);
constinit std::hive<int> constant_with_limits_and_allocator(constant_limits, std::allocator<int>());

template <class Alloc>
void test() {
  using H = std::hive<typename Alloc::value_type, Alloc>;

  const std::hive_limits hard = H::block_capacity_hard_limits();
  const std::hive_limits narrow(hard.min, hard.min);
  const std::hive_limits wide(hard.min, hard.max);
  const std::hive_limits mid(hard.min + (hard.max - hard.min) / 2, hard.max);
  const std::hive_limits top(hard.max, hard.max);

  for (std::hive_limits l : {narrow, wide, mid, top}) {
    {
      H h(l);
      check_empty(h, l);
    }
    {
      H h(l, Alloc());
      check_empty(h, l);
      assert(h.get_allocator() == Alloc());
    }
    {
      H h = {l, Alloc()};
      check_empty(h, l);
    }
  }
}

void test_stateful_allocator() {
  using H = std::hive<int, test_allocator<int>>;

  const std::hive_limits l = H::block_capacity_default_limits();
  H h(l, test_allocator<int>(7));
  check_empty(h, l);
  assert(h.get_allocator().get_data() == 7);
}

void test_constant_initialized() {
  check_empty(constant_with_limits, constant_limits);
  check_empty(constant_with_limits_and_allocator, constant_limits);
}

int main(int, char**) {
  test<std::allocator<int>>();
  test<std::allocator<char>>();
  test<min_allocator<int>>();
  test<test_allocator<int>>();
  test_stateful_allocator();
  test_constant_initialized();

  return 0;
}
