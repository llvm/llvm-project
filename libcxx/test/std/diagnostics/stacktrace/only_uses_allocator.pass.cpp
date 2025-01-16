//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23
// ADDITIONAL_COMPILE_FLAGS: -O0 -g
// UNSUPPORTED: asan, msan, tsan, hwasan, sanitizer-new-delete
// XFAIL: availability-stacktrace-missing

#include <cassert>
#include <cstddef>
#include <cstdlib>
#include <memory>
#include <stacktrace>

#include "test_allocs.h"

/*
 * This file includes tests which ensure any allocations performed by `basic_stacktrace`
 * are done via the user-provided allocator.  We intercept the usual ways to allocate,
 * counting the number of calls, through and not through the allocator.
 * (This won't work properly with sanitizers, hence the `UNSUPPORTED` above.)
 */

unsigned new_count      = 0;
unsigned del_count      = 0;
unsigned custom_alloc   = 0;
unsigned custom_dealloc = 0;

void* operator new(size_t sz) {
  ++new_count;
  auto* ret = malloc(sz);
  return ret;
}

void* operator new[](size_t sz) {
  ++new_count;
  auto* ret = malloc(sz);
  return ret;
}

void operator delete(void* ptr) noexcept {
  ++del_count;
  free(ptr);
}

void operator delete(void* ptr, size_t) noexcept {
  ++del_count;
  free(ptr);
}

void operator delete[](void* ptr) noexcept {
  ++del_count;
  free(ptr);
}

void operator delete[](void* ptr, size_t) noexcept {
  ++del_count;
  free(ptr);
}

template <typename T>
struct test_alloc : std::allocator<T> {
  using base = std::allocator<T>;

  template <typename U>
  struct rebind {
    using other = test_alloc<U>;
  };

  T* allocate(size_t n) {
    ++custom_alloc;
    auto* ret = base::allocate(n);
    return ret;
  }

  std::allocation_result<T*, size_t> allocate_at_least(size_t n) {
    ++custom_alloc;
    auto ret = base::allocate_at_least(n);
    return ret;
  }

  void deallocate(T* p, size_t n) {
    ++custom_dealloc;
    base::deallocate(p, n);
  }
};

_LIBCPP_NO_TAIL_CALLS
int main(int, char**) {
  (void)std::stacktrace::current();

  // Clear these counters in case anything was created/deleted prior to `main`,
  // and in case taking a stacktrace involved initialization of objects in bss
  // or some other space.
  new_count = del_count = 0;

  {
    using A = test_alloc<std::stacktrace_entry>;
    A alloc;
    auto st = std::basic_stacktrace<A>::current(alloc);
    // Ensure allocator was called at some point
    assert(custom_alloc > 0);
    // Exit this scope to destroy stacktrace and allocator
  }

  assert(custom_alloc == new_count);      // All objects should have come from allocator,
  assert(custom_alloc == custom_dealloc); // and all allocations should be deallocated

  return 0;
}
