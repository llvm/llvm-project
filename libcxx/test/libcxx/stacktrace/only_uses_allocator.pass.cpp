//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23
// UNSUPPORTED: asan, msan, tsan, hwasan, sanitizer-new-delete
// ADDITIONAL_COMPILE_FLAGS: -O0 -g

#include <cassert>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <stacktrace>

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
  std::cerr << "op new: " << new_count << ": new:   " << ret << " size " << sz << '\n';
  return ret;
}

void* operator new[](size_t sz) {
  ++new_count;
  auto* ret = malloc(sz);
  std::cerr << "op new: " << new_count << ": new[]: " << ret << " size " << sz << '\n';
  return ret;
}

void operator delete(void* ptr) noexcept {
  ++del_count;
  std::cerr << "op del: " << del_count << ": del:   " << ptr << '\n';
  free(ptr);
}

void operator delete(void* ptr, size_t sz) noexcept {
  ++del_count;
  std::cerr << "op del: " << del_count << ": del:   " << ptr << " size " << sz << '\n';
  free(ptr);
}

void operator delete[](void* ptr) noexcept {
  ++del_count;
  std::cerr << "op del: " << del_count << ": del[]: " << ptr << '\n';
  free(ptr);
}

void operator delete[](void* ptr, size_t sz) noexcept {
  ++del_count;
  std::cerr << "op del: " << del_count << ": del[]: " << ptr << " size " << sz << '\n';
  free(ptr);
}

template <typename T>
struct test_alloc {
  using size_type     = size_t;
  using value_type    = T;
  using pointer       = T*;
  using const_pointer = T const*;

  template <typename U>
  struct rebind {
    using other = test_alloc<U>;
  };

  test_alloc() = default;

  template <typename U = T>
  test_alloc(const test_alloc<U>&) {}

  bool operator==(auto const& rhs) const { return &rhs == this; }
  bool operator==(test_alloc const&) const { return true; }

  T* allocate(size_t n) {
    ++custom_alloc;
    return new T[n];
  }

  std::allocation_result<T*> allocate_at_least(size_t n) { return {.ptr = allocate(n), .count = n}; }

  void deallocate(T*, size_t) { ++custom_dealloc; }
};

_LIBCPP_NO_TAIL_CALLS
int main(int, char**) {
  std::cerr << "initial call to `current`\n";
  (void)std::stacktrace::current();

  // Clear these counters in case anything was created/deleted prior to `main`,
  // and in case taking a stacktrace involved initialization of objects in bss
  // or some other space.
  new_count = del_count = 0;

  {
    using A = test_alloc<std::stacktrace_entry>;
    std::cerr << "calling `current` with allocator\n";
    A alloc;
    auto st = std::basic_stacktrace<A>::current(alloc);
    // Ensure allocator was called at some point
    assert(custom_alloc > 0);
    // Exit this scope to destroy stacktrace and allocator
  }

  assert(custom_alloc == new_count);      // All "new" calls should have been through allocator
  assert(custom_alloc == custom_dealloc); // and all allocations should be deallocated

  return 0;
}
