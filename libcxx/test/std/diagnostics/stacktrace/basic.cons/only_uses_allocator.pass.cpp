//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23
// ADDITIONAL_COMPILE_FLAGS: -g -O0

#include <cassert>
#include <cstdlib>
#include <stacktrace>

/*
 * This file includes tests which ensure any allocations performed by `basic_stacktrace`
 * are done via the user-provided allocator.  We intercept the usual ways to allocate,
 * counting the number of calls, through and not through the allocator.
 */

unsigned new_count;
unsigned del_count;
unsigned custom_alloc;
unsigned custom_dealloc;

void* operator new(size_t size) {
  ++new_count;
  return malloc(size);
}
void* operator new[](size_t size) {
  ++new_count;
  return malloc(size);
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
struct test_alloc {
  using size_type     = size_t;
  using value_type    = T;
  using pointer       = T*;
  using const_pointer = T const*;

  template <typename U>
  struct rebind {
    using other = test_alloc<U>;
  };

  std::allocator<T> wrapped_{};

  test_alloc() = default;

  template <typename U>
  test_alloc(test_alloc<U> const& rhs) : wrapped_(rhs.wrapped_) {}

  bool operator==(auto const& rhs) const { return &rhs == this; }
  bool operator==(test_alloc const&) const { return true; }

  T* allocate(size_t n) {
    ++custom_alloc;
    return wrapped_.allocate(n);
  }

  auto allocate_at_least(size_t n) {
    ++custom_alloc;
    return wrapped_.allocate_at_least(n);
  }

  void deallocate(T* ptr, size_t n) {
    ++custom_dealloc;
    wrapped_.deallocate(ptr, n);
  }
};

_LIBCPP_NO_TAIL_CALLS
int main(int, char**) {
  // Clear these counters in case anything was created/deleted prior to `main`,
  // and in case taking a stacktrace involved initialization of something and is
  // outside our control.
  (void)std::stacktrace::current();
  new_count = del_count = 0;

  {
    using A = test_alloc<std::stacktrace_entry>;
    auto st = std::basic_stacktrace<A>::current();
    assert(custom_alloc > 0); // Ensure allocator was called at some point
  } // Exit this scope to destroy stacktrace (as well as allocator)

  assert(custom_alloc == new_count);      // All "new" calls should have been through allocator
  assert(custom_alloc == custom_dealloc); // and all allocations should be deallocated

  return 0;
}
