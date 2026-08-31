//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23
// UNSUPPORTED: availability-stacktrace-missing

// (19.6.4.2): [stacktrace.basic.cons], creation and assignment
//
//   basic_stacktrace() noexcept(is_nothrow_default_constructible_v<allocator_type>);
//
//   explicit basic_stacktrace(const allocator_type& alloc) noexcept;

#include <cassert>
#include <memory>
#include <stacktrace>
#include <type_traits>

namespace {

template <typename T>
struct AllocCtorCanThrow : std::allocator<T> {
  AllocCtorCanThrow() noexcept(false) : std::allocator<T>{} {}
};

template <typename T>
struct AllocCtorNoThrow : std::allocator<T> {
  AllocCtorNoThrow() noexcept : std::allocator<T>{} {}
};

static_assert(!std::is_nothrow_default_constructible_v<AllocCtorCanThrow<int>>);
static_assert(std::is_nothrow_default_constructible_v<AllocCtorNoThrow<int>>);

} // namespace

int main(int, char**) {
  {
    using A = AllocCtorCanThrow<std::stacktrace_entry>;
    static_assert(!noexcept(std::basic_stacktrace<A>()));
    std::basic_stacktrace<A> st;
    assert(st.empty()); // Postconditions: empty() is true.
  }

  {
    using A = AllocCtorNoThrow<std::stacktrace_entry>;
    static_assert(noexcept(std::basic_stacktrace<A>()));
    std::basic_stacktrace<A> st;
    assert(st.empty()); // Postconditions: empty() is true.
  }

  {
    using A = AllocCtorCanThrow<std::stacktrace_entry>;
    A a;
    static_assert(noexcept(std::basic_stacktrace<A>{a}));
    std::basic_stacktrace<A> st{a};
    assert(st.empty()); // Postconditions: empty() is true.
  }

  return 0;
}
