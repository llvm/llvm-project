//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: no-exceptions

// <unordered_map>

// Check that the unordered_map copy constructor does not leak nodes when a node allocation
// throws partway through copying the elements. The already-copied nodes must be released even
// though the container itself never finishes constructing.
//
// Also check the copy-assignment-into-empty path, which shares the same node-copying helper.
// There the assigned-to container stays alive, so after a throw it must be left in a valid
// (usable, leak-free) state with no dangling bucket pointers.

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
  typedef T value_type;

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
  typedef CountedThrowingAllocator<std::pair<const int, int> > Alloc;
  typedef std::unordered_map<int, int, std::hash<int>, std::equal_to<int>, Alloc> Map;

  const int never_throw = 1 << 30;
  int countdown         = never_throw;

  // Build a source map large enough that copying it performs several node allocations. The
  // allocator is propagated by select_on_container_copy_construction, so the copy shares this
  // same countdown.
  Map src((Alloc(countdown)));
  for (int i = 0; i < 16; ++i)
    src.insert(std::make_pair(i, i));

  // Make a node allocation fail at every position partway through the copy and verify that no
  // nodes are leaked. `outstanding_before` captures the allocations held by the source so the
  // check is unaffected by it.
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

  // Same, but through copy-assignment into an empty (default-constructed) map. The destination
  // survives the exception, so it must be left valid: we exercise it with insert()/find() (which
  // read the bucket array and would dereference a stale entry under a sanitizer) and confirm no
  // nodes are leaked once it goes out of scope.
  for (int fail_at = 0; fail_at < 32; ++fail_at) {
    const int outstanding_before = globalMemCounter.outstanding_new;
    {
      Map dst((Alloc(countdown)));
      countdown = fail_at;
      try {
        dst = src;
      } catch (const std::bad_alloc&) {
      }
      countdown = never_throw;

      // The container must be in a valid state whether or not the assignment threw.
      for (int i = 0; i < 16; ++i)
        dst.insert(std::make_pair(i, i));
      for (int i = 0; i < 16; ++i)
        assert(dst.find(i) != dst.end());
    }
    assert(globalMemCounter.outstanding_new == outstanding_before);
  }

  return 0;
}
