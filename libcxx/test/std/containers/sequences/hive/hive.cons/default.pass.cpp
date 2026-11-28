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

// constexpr hive() noexcept(noexcept(Allocator()));
// constexpr explicit hive(const Allocator&) noexcept;

#include <cassert>
#include <cstddef>
#include <hive>
#include <memory>
#include <type_traits>

#include "../helpers.h"
#include "min_allocator.h"
#include "test_allocator.h"
#include "test_macros.h"

template <class T>
struct MayThrowDefaultAllocator {
  using value_type = T;

  MayThrowDefaultAllocator() noexcept(false) {}
  template <class U>
  MayThrowDefaultAllocator(const MayThrowDefaultAllocator<U>&) noexcept {}

  T* allocate(std::size_t n) { return std::allocator<T>().allocate(n); }
  void deallocate(T* p, std::size_t n) noexcept { std::allocator<T>().deallocate(p, n); }

  friend bool operator==(const MayThrowDefaultAllocator&, const MayThrowDefaultAllocator&) = default;
};

static_assert(std::is_nothrow_default_constructible_v<std::hive<int>>);
static_assert(!std::is_nothrow_default_constructible_v<std::hive<int, MayThrowDefaultAllocator<int>>>);
static_assert(std::is_nothrow_constructible_v<std::hive<int>, const std::allocator<int>&>);
static_assert(std::is_nothrow_constructible_v<std::hive<int, MayThrowDefaultAllocator<int>>,
                                              const MayThrowDefaultAllocator<int>&>);
static_assert(!std::is_convertible_v<const std::allocator<int>&, std::hive<int>>);

constexpr std::allocator<int> constant_allocator;
constinit std::hive<int> constant_default;
constinit std::hive<int> constant_with_allocator(constant_allocator);

template <class Alloc>
void test() {
  using H = std::hive<typename Alloc::value_type, Alloc>;

  {
    H h;
    check_empty(h, H::block_capacity_default_limits());
    assert(h.get_allocator() == Alloc());
  }
  {
    H h = {};
    check_empty(h, H::block_capacity_default_limits());
  }
  {
    const Alloc a{};
    H h(a);
    check_empty(h, H::block_capacity_default_limits());
    assert(h.get_allocator() == a);
  }
}

void test_stateful_allocator() {
  using H = std::hive<int, test_allocator<int>>;

  test_allocator_statistics stats;
  H h(test_allocator<int>(42, &stats));
  check_empty(h, H::block_capacity_default_limits());
  assert(h.get_allocator().get_data() == 42);
  assert(stats.alloc_count == 0);
}

void test_constant_initialized() {
  check_empty(constant_default, std::hive<int>::block_capacity_default_limits());
  check_empty(constant_with_allocator, std::hive<int>::block_capacity_default_limits());
}

int main(int, char**) {
  test<std::allocator<int>>();
  test<std::allocator<double>>();
  test<min_allocator<int>>();
  test<test_allocator<int>>();
  test<MayThrowDefaultAllocator<int>>();
  test_stateful_allocator();
  test_constant_initialized();

  return 0;
}
