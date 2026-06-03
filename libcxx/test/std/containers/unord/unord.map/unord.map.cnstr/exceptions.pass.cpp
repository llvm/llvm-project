//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: no-exceptions

// <unordered_map>

// Check that the unordered_map copy constructor does not leak nodes when an allocation
// throws partway through copying the elements. The enclosing container is not fully
// constructed yet, so its destructor does not run to release the already-copied nodes.

#include <cassert>
#include <cstddef>
#include <functional>
#include <memory>
#include <new>
#include <unordered_map>
#include <utility>

#include "count_new.h"
#include "test_macros.h"

// An allocator that throws std::bad_alloc once a shared countdown reaches zero. It forwards
// to std::allocator (operator new) so that count_new.h can observe leaked nodes.
template <class T>
struct CountedThrowingAllocator {
  using value_type = T;

  int* countdown_;

  explicit CountedThrowingAllocator(int& countdown) : countdown_(&countdown) {}

  template <class U>
  CountedThrowingAllocator(const CountedThrowingAllocator<U>& other) : countdown_(other.countdown_) {}

  T* allocate(std::size_t n) {
    if (*countdown_ == 0)
      throw std::bad_alloc();
    --*countdown_;
    return std::allocator<T>().allocate(n);
  }

  void deallocate(T* p, std::size_t n) { std::allocator<T>().deallocate(p, n); }

  template <class U>
  friend bool operator==(const CountedThrowingAllocator&, const CountedThrowingAllocator<U>&) {
    return true;
  }

  template <class U>
  friend bool operator!=(const CountedThrowingAllocator&, const CountedThrowingAllocator<U>&) {
    return false;
  }
};

int main(int, char**) {
  using Alloc = CountedThrowingAllocator<std::pair<const int, int> >;
  using Map   = std::unordered_map<int, int, std::hash<int>, std::equal_to<int>, Alloc>;

  const int never_throw = 1 << 30;
  int countdown         = never_throw;

  // Build a source map large enough that copying it performs several node allocations.
  Map src((Alloc(countdown)));
  for (int i = 0; i < 16; ++i)
    src.emplace(i, i);

  // For every point at which an allocation can fail during the copy, verify that the
  // partially-constructed copy does not leak any nodes. The allocator is propagated by
  // select_on_container_copy_construction, so it shares the same countdown.
  for (int fail_at = 0; fail_at < 32; ++fail_at) {
    const int outstanding_before = globalMemCounter.outstanding_new;

    countdown = fail_at;
    try {
      Map copy(src);
      (void)copy;
    } catch (const std::bad_alloc&) {
    }
    countdown = never_throw;

    assert(globalMemCounter.outstanding_new == outstanding_before);
  }

  return 0;
}
