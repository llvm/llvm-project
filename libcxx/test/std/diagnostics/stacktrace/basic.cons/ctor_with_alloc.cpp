//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23
// ADDITIONAL_COMPILE_FLAGS: -g

/*
  (19.6.4.2)

  // [stacktrace.basic.cons], creation and assignment
  explicit basic_stacktrace(const allocator_type& alloc) noexcept;
*/

#include <cassert>
#include <stacktrace>

uint32_t test1_line;
uint32_t test2_line;

template <class A>
_LIBCPP_NO_TAIL_CALLS _LIBCPP_NOINLINE std::basic_stacktrace<A> test1(A& alloc) {
  test1_line = __LINE__ + 1; // add 1 to get the next line (where the call to `current` occurs)
  auto ret   = std::basic_stacktrace<A>::current(alloc);
  return ret;
}

template <class A>
_LIBCPP_NO_TAIL_CALLS _LIBCPP_NOINLINE std::basic_stacktrace<A> test2(A& alloc) {
  test2_line = __LINE__ + 1; // add 1 to get the next line (where the call to `current` occurs)
  auto ret   = test1(alloc);
  return ret;
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

  T* allocate(size_t n) { return wrapped_.allocate(n); }
  auto allocate_at_least(size_t n) { return wrapped_.allocate_at_least(n); }
  void deallocate(T* ptr, size_t n) { return wrapped_.deallocate(ptr, n); }
};

_LIBCPP_NO_TAIL_CALLS _LIBCPP_NOINLINE void test_construct_with_allocator() {
  test_alloc<std::stacktrace_entry> alloc;
  std::basic_stacktrace<decltype(alloc)> st(alloc);
  assert(st.empty());

  st = std::basic_stacktrace<decltype(alloc)>::current(alloc);
  assert(!st.empty());
}

_LIBCPP_NO_TAIL_CALLS
int main(int, char**) {
  test_construct_with_allocator();
  return 0;
}
