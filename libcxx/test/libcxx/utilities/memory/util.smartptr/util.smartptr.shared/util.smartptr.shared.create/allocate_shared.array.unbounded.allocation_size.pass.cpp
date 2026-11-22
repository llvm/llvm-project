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

#include "test_macros.h"

struct AllocationStats {
  std::size_t allocations = 0;
  std::size_t bytes       = 0;
};

template <class T>
class ByteCountingAllocator {
  AllocationStats* stats_ = nullptr;

  template <class U>
  friend class ByteCountingAllocator;

public:
  using value_type = T;

  explicit ByteCountingAllocator(AllocationStats* stats) : stats_(stats) {}

  template <class U>
  ByteCountingAllocator(const ByteCountingAllocator<U>& other) noexcept : stats_(other.stats_) {}

  T* allocate(std::size_t n) {
    ++stats_->allocations;
    stats_->bytes += n * sizeof(T);
    return std::allocator<T>().allocate(n);
  }

  void deallocate(T* p, std::size_t n) noexcept {
    --stats_->allocations;
    stats_->bytes -= n * sizeof(T);
    std::allocator<T>().deallocate(p, n);
  }

  template <class U>
  friend bool operator==(const ByteCountingAllocator& a, const ByteCountingAllocator<U>& b) noexcept {
    return a.stats_ == b.stats_;
  }
};

template <class Array>
void test(std::size_t n) {
  using Element                    = std::remove_extent_t<Array>;
  constexpr std::size_t max_header = 256;
  const std::size_t min_bytes      = n * sizeof(Element);
  const std::size_t max_bytes      = min_bytes + max_header;

  AllocationStats stats;
  {
    std::shared_ptr<Array> ptr = std::allocate_shared<Array>(ByteCountingAllocator<Array>(&stats), n);
    DoNotOptimize(ptr.get());
    assert(stats.allocations == 1);
    assert(stats.bytes >= min_bytes);
    assert(stats.bytes <= max_bytes);
  }
  assert(stats.allocations == 0);
  assert(stats.bytes == 0);

  {
    Element init{};
    std::shared_ptr<Array> ptr = std::allocate_shared<Array>(ByteCountingAllocator<Array>(&stats), n, init);
    DoNotOptimize(ptr.get());
    assert(stats.allocations == 1);
    assert(stats.bytes >= min_bytes);
    assert(stats.bytes <= max_bytes);
  }
  assert(stats.allocations == 0);
  assert(stats.bytes == 0);

  {
    std::shared_ptr<Array> ptr = std::allocate_shared_for_overwrite<Array>(ByteCountingAllocator<Array>(&stats), n);
    DoNotOptimize(ptr.get());
    assert(stats.allocations == 1);
    assert(stats.bytes >= min_bytes);
    assert(stats.bytes <= max_bytes);
  }
  assert(stats.allocations == 0);
  assert(stats.bytes == 0);
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
