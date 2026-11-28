//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// <memory>

// shared_ptr

// template<class T, class A>
// shared_ptr<T> allocate_shared(const A& a, size_t N); // T is U[]
//
// template<class T, class A>
// shared_ptr<T> allocate_shared(const A& a, size_t N, const remove_extent_t<T>& u); // T is U[]
//
// template<class T, class A>
// shared_ptr<T> allocate_shared_for_overwrite(const A& a, size_t N); // T is U[]

// Check that we don't over-allocate for unbounded arrays.

#include <cassert>
#include <cstddef>
#include <memory>
#include <type_traits>

#include "test_allocator.h"
#include "test_macros.h"

template <class Array>
void test(std::size_t n) {
  using Element                    = std::remove_extent_t<Array>;
  constexpr std::size_t max_header = 256;
  const std::size_t min_bytes      = n * sizeof(Element);
  const std::size_t max_bytes      = min_bytes + max_header;

  test_allocator_statistics stats;
  {
    std::shared_ptr<Array> ptr = std::allocate_shared<Array>(test_allocator<void>(&stats), n);
    DoNotOptimize(ptr.get());
    assert(stats.alloc_count == 1);
    assert(stats.allocated_bytes >= min_bytes);
    assert(stats.allocated_bytes <= max_bytes);
  }
  assert(stats.alloc_count == 0);
  assert(stats.allocated_bytes == 0);

  {
    Element init{};
    std::shared_ptr<Array> ptr = std::allocate_shared<Array>(test_allocator<void>(&stats), n, init);
    DoNotOptimize(ptr.get());
    assert(stats.alloc_count == 1);
    assert(stats.allocated_bytes >= min_bytes);
    assert(stats.allocated_bytes <= max_bytes);
  }
  assert(stats.alloc_count == 0);
  assert(stats.allocated_bytes == 0);

  {
    std::shared_ptr<Array> ptr = std::allocate_shared_for_overwrite<Array>(test_allocator<void>(&stats), n);
    DoNotOptimize(ptr.get());
    assert(stats.alloc_count == 1);
    assert(stats.allocated_bytes >= min_bytes);
    assert(stats.allocated_bytes <= max_bytes);
  }
  assert(stats.alloc_count == 0);
  assert(stats.allocated_bytes == 0);
}

int main(int, char**) {
  const std::size_t sizes[] = {0, 1, 1024};
  for (std::size_t n : sizes) {
    test<char[]>(n);
    test<int[]>(n);
    test<char[][4]>(n);
  }

  return 0;
}
