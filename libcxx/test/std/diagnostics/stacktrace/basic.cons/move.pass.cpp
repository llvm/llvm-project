//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23
// UNSUPPORTED: availability-stacktrace-missing

// (19.6.4.2) [stacktrace.basic.cons], creation and assignment
//
//   basic_stacktrace(basic_stacktrace&& other) noexcept;
//
//   basic_stacktrace(basic_stacktrace&& other, const allocator_type& alloc);
//
//   basic_stacktrace& operator=(basic_stacktrace&& other)
//     noexcept(allocator_traits<Allocator>::propagate_on_container_move_assignment::value ||
//       allocator_traits<Allocator>::is_always_equal::value);

#include <cassert>
#include <stacktrace>
#include <type_traits>
#include <utility>

#include "test_allocator.h"
#include "test_macros.h"

// The standard permits self-move-assignment to leave the object in any valid-but-unspecified
// state; we only assert it doesn't crash/UB. Silence the warning so we can test it.
TEST_CLANG_DIAGNOSTIC_IGNORED("-Wself-move")

namespace {

template <typename T>
struct AllocNoPropagate : std::allocator<T> {
  using propagate_on_container_move_assignment = std::false_type;
  using is_always_equal                        = std::false_type;
};

template <typename T>
struct AllocPropagate : std::allocator<T> {
  using propagate_on_container_move_assignment = std::true_type;
  using is_always_equal                        = std::false_type;
};

template <typename T>
struct AllocAlwaysEqual : std::allocator<T> {
  using propagate_on_container_move_assignment = std::false_type;
  using is_always_equal                        = std::true_type;
};

} // namespace

int main() {
  // Move-construction tests

  {
    using A = AllocNoPropagate<std::stacktrace_entry>;
    auto s0 = std::basic_stacktrace<A>::current();
    std::basic_stacktrace<A> s1{s0};
    static_assert(noexcept(std::basic_stacktrace<A>(std::move(s0))));
    std::basic_stacktrace<A> s2(std::move(s0));
    assert(s1 == s2);
  }

  {
    using A = AllocPropagate<std::stacktrace_entry>;
    auto s0 = std::basic_stacktrace<A>::current();
    std::basic_stacktrace<A> s1{s0};
    static_assert(noexcept(std::basic_stacktrace<A>(std::move(s0))));
    std::basic_stacktrace<A> s2(std::move(s0));
  }

  {
    using A = AllocAlwaysEqual<std::stacktrace_entry>;
    auto s0 = std::basic_stacktrace<A>::current();
    std::basic_stacktrace<A> s1{s0};
    static_assert(noexcept(std::basic_stacktrace<A>(std::move(s0))));
    std::basic_stacktrace<A> s2(std::move(s0));
  }

  // Move-construction (without an explicit allocator) must take over `other`'s allocator
  // unconditionally -- regardless of propagate_on_container_move_assignment or is_always_equal,
  // which only govern move-*assignment*, not move-*construction*.
  {
    using A = test_allocator<std::stacktrace_entry>;
    A alloc(42);
    std::basic_stacktrace<A> s0(alloc);
    std::basic_stacktrace<A> s1(std::move(s0));
    assert(s1.get_allocator().get_data() == 42);
  }

  // Move-construction with an explicit allocator

  {
    using A = std::allocator<std::stacktrace_entry>;
    auto s0 = std::basic_stacktrace<A>::current();
    std::basic_stacktrace<A> s1{s0};
    std::basic_stacktrace<A> s2{std::move(s0), A()};
    assert(s1 == s2);
  }

  // Move-assignment tests

  {
    using A = AllocNoPropagate<std::stacktrace_entry>;
    auto s0 = std::basic_stacktrace<A>::current();
    std::basic_stacktrace<A> s1{s0};
    std::basic_stacktrace<A> s2;
    static_assert(!noexcept(s2 = std::move(s0)));
    s2 = std::move(s0);
    assert(s1 == s2);
  }

  {
    using A = AllocPropagate<std::stacktrace_entry>;
    auto s0 = std::basic_stacktrace<A>::current();
    std::basic_stacktrace<A> s1{s0};
    std::basic_stacktrace<A> s2;
    static_assert(noexcept(s2 = std::move(s0)));
    s2 = std::move(s0);
    assert(s1 == s2);
  }

  {
    using A = AllocAlwaysEqual<std::stacktrace_entry>;
    auto s0 = std::basic_stacktrace<A>::current();
    std::basic_stacktrace<A> s1{s0};
    std::basic_stacktrace<A> s2;
    static_assert(noexcept(s2 = std::move(s0)));
    s2 = std::move(s0);
    assert(s1 == s2);
  }

  // Self-move-assignment must not crash; the standard leaves the resulting
  // state valid-but-unspecified, so we only check it's still a usable object.

  {
    auto s0 = std::stacktrace::current();
    s0      = std::move(s0);
    assert(s0.size() <= s0.max_size());
  }

  return 0;
}
