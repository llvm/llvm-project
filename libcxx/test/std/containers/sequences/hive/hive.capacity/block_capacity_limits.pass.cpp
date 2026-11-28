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

// constexpr hive_limits block_capacity_limits() const noexcept;
// static constexpr hive_limits block_capacity_default_limits() noexcept;
// static constexpr hive_limits block_capacity_hard_limits() noexcept;
// static constexpr bool is_within_hard_limits(hive_limits) noexcept;

#include <algorithm>
#include <cassert>
#include <concepts>
#include <cstdint>
#include <hive>
#include <memory>
#include <utility>
#include <vector>

#include "min_allocator.h"
#include "sized_allocator.h"
#include "test_macros.h"

template <class H>
constexpr bool test_static_limits() {
  ASSERT_NOEXCEPT(H::block_capacity_default_limits());
  ASSERT_NOEXCEPT(H::block_capacity_hard_limits());
  ASSERT_NOEXCEPT(H::is_within_hard_limits(std::hive_limits(1, 1)));

  std::same_as<std::hive_limits> decltype(auto) hard = H::block_capacity_hard_limits();
  std::same_as<std::hive_limits> decltype(auto) def  = H::block_capacity_default_limits();
  std::same_as<bool> decltype(auto) hard_is_within   = H::is_within_hard_limits(hard);

  // The hard limits bound the defaults.
  assert(hard.min <= hard.max);
  assert(def.min <= def.max);
  assert(hard.min <= def.min);
  assert(def.max <= hard.max);
  assert(hard_is_within);
  assert(H::is_within_hard_limits(def));

  // [hive.overview]: the maximum hard limit is no larger than allocator_traits<Allocator>::max_size().
  using A = typename H::allocator_type;
  assert(hard.max <= std::allocator_traits<A>::max_size(A()));

  // Outside the hard limits, or min > max.
  assert(!H::is_within_hard_limits(std::hive_limits(hard.max, hard.min)) || hard.min == hard.max);
  assert(!H::is_within_hard_limits(std::hive_limits(hard.min, hard.max + 1)));
  if (hard.min > 0) {
    assert(!H::is_within_hard_limits(std::hive_limits(hard.min - 1, hard.max)));
  }
  if (hard.min < hard.max) {
    assert(!H::is_within_hard_limits(std::hive_limits(hard.min + 1, hard.min)));
  }

  return true;
}

template <class H>
void test_current_limits() {
  ASSERT_NOEXCEPT(std::declval<const H&>().block_capacity_limits());

  const H h;
  std::same_as<std::hive_limits> decltype(auto) l = h.block_capacity_limits();
  const std::hive_limits def                      = H::block_capacity_default_limits();
  assert(l.min == def.min);
  assert(l.max == def.max);
}

template <class H>
void test() {
  test_static_limits<H>();
  static_assert(test_static_limits<H>());
  test_current_limits<H>();
}

// With a size_type narrower than size_t, blocks at the maximum hard limit still hold all of their elements.
void test_narrow_size_type() {
  using H = std::hive<int, sized_allocator<int, std::uint8_t, std::int8_t>>;

  const std::hive_limits hard = H::block_capacity_hard_limits();
  H h(std::hive_limits(hard.max, hard.max));
  std::vector<int> expected;
  for (int i = 0; i != static_cast<int>(hard.max) + 1; ++i) {
    h.insert(i);
    expected.push_back(i);
  }
  assert(h.size() == hard.max + 1);
  assert(h.capacity() >= hard.max + 1);

  std::vector<int> seen(h.begin(), h.end());
  std::sort(seen.begin(), seen.end());
  assert(seen == expected);
}

struct Big {
  char data[1000];
};

int main(int, char**) {
  test<std::hive<int>>();
  test<std::hive<char>>();
  test<std::hive<Big>>();
  test<std::hive<int, min_allocator<int>>>();
  test<std::hive<int, sized_allocator<int, std::uint8_t, std::int8_t>>>();
  test<std::hive<Big, sized_allocator<Big, std::uint16_t, std::int16_t>>>();
  test_narrow_size_type();

  return 0;
}
