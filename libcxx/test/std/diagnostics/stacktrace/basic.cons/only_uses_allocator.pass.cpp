//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23
// ADDITIONAL_COMPILE_FLAGS: -g -O0

/*
  (19.6.4.2)

  // [stacktrace.basic.cons], creation and assignment
  static basic_stacktrace current(const allocator_type& alloc = allocator_type()) noexcept;   [1]
  static basic_stacktrace current(size_type skip,
                                  const allocator_type& alloc = allocator_type()) noexcept;   [2]
  static basic_stacktrace current(size_type skip, size_type max_depth,
                                  const allocator_type& alloc = allocator_type()) noexcept;   [3]

  basic_stacktrace() noexcept(is_nothrow_default_constructible_v<allocator_type>);            [4]
  explicit basic_stacktrace(const allocator_type& alloc) noexcept;                            [5]

  basic_stacktrace(const basic_stacktrace& other);                                            [6]
  basic_stacktrace(basic_stacktrace&& other) noexcept;                                        [7]
  basic_stacktrace(const basic_stacktrace& other, const allocator_type& alloc);               [8]
  basic_stacktrace(basic_stacktrace&& other, const allocator_type& alloc);                    [9]
  basic_stacktrace& operator=(const basic_stacktrace& other);                                 [10]
  basic_stacktrace& operator=(basic_stacktrace&& other)
    noexcept(allocator_traits<Allocator>::propagate_on_container_move_assignment::value ||
      allocator_traits<Allocator>::is_always_equal::value);                                   [11]

  ~basic_stacktrace();                                                                        [12]
*/

#include <cassert>
#include <cstdlib>
#include <stacktrace>

/*
 * This file includes tests which ensure any allocations performed by `basic_stacktrace`
 * are done via the user-provided allocator.  We intercept the usual ways to allocate,
 * counting the number of calls.
 */

unsigned custom_alloc;
unsigned custom_dealloc;

void* operator new(size_t size) { return malloc(size); }
void* operator new[](size_t size) { return malloc(size); }
void operator delete(void* ptr) noexcept { free(ptr); }
void operator delete(void* ptr, size_t) noexcept { free(ptr); }
void operator delete[](void* ptr) noexcept { free(ptr); }
void operator delete[](void* ptr, size_t) noexcept { free(ptr); }

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

/*
  (19.6.4.2) [stacktrace.basic.cons], creation and assignment,
  only exercising usage of caller-provided allocator.

  static basic_stacktrace current(const allocator_type& alloc = allocator_type()) noexcept;   [1]
  static basic_stacktrace current(size_type skip,
                                  const allocator_type& alloc = allocator_type()) noexcept;   [2]
  static basic_stacktrace current(size_type skip, size_type max_depth,
                                  const allocator_type& alloc = allocator_type()) noexcept;   [3]

  explicit basic_stacktrace(const allocator_type& alloc) noexcept;                            [5]
  basic_stacktrace(const basic_stacktrace& other, const allocator_type& alloc);               [8]
  basic_stacktrace(basic_stacktrace&& other, const allocator_type& alloc);                    [9]

  basic_stacktrace& operator=(const basic_stacktrace& other);                                 [10]
  basic_stacktrace& operator=(basic_stacktrace&& other)
    noexcept(allocator_traits<Allocator>::propagate_on_container_move_assignment::value ||
      allocator_traits<Allocator>::is_always_equal::value);                                   [11]
*/

void do_current_stacktrace() {
  using A = test_alloc<std::stacktrace_entry>;
  (void)std::basic_stacktrace<A>::current(A());
}

_LIBCPP_NO_TAIL_CALLS
int main(int, char**) {
  {
    do_current_stacktrace();
    assert(custom_alloc > 0);
  }
  assert(custom_dealloc == custom_alloc);
  return 0;
}
