//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23
// XFAIL: availability-stacktrace-missing

/*
  (19.6.4.5) Modifiers [stacktrace.basic.mod]

  template<class Allocator>
  void swap(basic_stacktrace& other)
      noexcept(allocator_traits<Allocator>::propagate_on_container_swap::value ||
      allocator_traits<Allocator>::is_always_equal::value);

  Effects: Exchanges the contents of *this and other.
*/

#include <cassert>
#include <stacktrace>

#include "../test_allocs.h"

int main(int, char**) {
  std::stacktrace empty;
  auto a = std::stacktrace::current();
  std::stacktrace b(empty);
  assert(!a.empty());
  assert(b.empty());

  a.swap(b);
  assert(a.empty());
  assert(!b.empty());

  // Check `noexcept`: `swap` is noexcept if either:
  //   (1) the allocator propagates on swap
  //   (2) if instances of that allocator type are always equal.

  // `AllocPropagate` satisfies the first (but not the second); stacktrace swap should be noexcept
  AllocPropagate<std::stacktrace_entry> prop1;
  AllocPropagate<std::stacktrace_entry> prop2;
  auto prop_st1 = std::basic_stacktrace<decltype(prop1)>(prop1);
  auto prop_st2 = std::basic_stacktrace<decltype(prop2)>(prop2);
  static_assert(noexcept(prop_st1.swap(prop_st2)));

  // `AllocNoPropagate` satisfies neither; stacktrace swap should not be noexcept
  AllocNoPropagate<std::stacktrace_entry> no_prop1;
  AllocNoPropagate<std::stacktrace_entry> no_prop2;
  auto no_prop_st1 = std::basic_stacktrace<decltype(no_prop1)>(no_prop1);
  auto no_prop_st2 = std::basic_stacktrace<decltype(no_prop2)>(no_prop2);
  static_assert(!noexcept(no_prop_st1.swap(no_prop_st2)));

  // `AllocAlwaysEqual` satisfies second; stacktrace swap should be noexcept
  AllocAlwaysEqual<std::stacktrace_entry> always_eq1;
  AllocAlwaysEqual<std::stacktrace_entry> always_eq2;
  auto always_eq_st1 = std::basic_stacktrace<decltype(always_eq1)>(always_eq1);
  auto always_eq_st2 = std::basic_stacktrace<decltype(always_eq2)>(always_eq2);
  static_assert(noexcept(always_eq_st1.swap(always_eq_st2)));

  return 0;
}
