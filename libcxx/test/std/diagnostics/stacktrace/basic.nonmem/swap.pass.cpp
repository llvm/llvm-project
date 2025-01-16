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
  (19.6.4.6) Non-member functions

  template<class Allocator>
    void swap(basic_stacktrace<Allocator>& a, basic_stacktrace<Allocator>& b)
      noexcept(noexcept(a.swap(b)));
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

  std::swap(a, b);
  assert(a.empty());
  assert(!b.empty());

  // `AllocPropagate` satisfies the first (but not the second); stacktrace swap should be noexcept
  AllocPropagate<std::stacktrace_entry> prop1;
  AllocPropagate<std::stacktrace_entry> prop2;
  auto prop_st1 = std::basic_stacktrace<decltype(prop1)>(prop1);
  auto prop_st2 = std::basic_stacktrace<decltype(prop2)>(prop2);
  static_assert(noexcept(std::swap(prop_st1, prop_st2)));

  // `AllocNoPropagate` satisfies neither; stacktrace swap should not be noexcept
  AllocNoPropagate<std::stacktrace_entry> no_prop1;
  AllocNoPropagate<std::stacktrace_entry> no_prop2;
  auto no_prop_st1 = std::basic_stacktrace<decltype(no_prop1)>(no_prop1);
  auto no_prop_st2 = std::basic_stacktrace<decltype(no_prop2)>(no_prop2);
  static_assert(!noexcept(std::swap(no_prop_st1, no_prop_st2)));

  // `AllocAlwaysEqual` satisfies second; stacktrace swap should be noexcept
  AllocAlwaysEqual<std::stacktrace_entry> always_eq1;
  AllocAlwaysEqual<std::stacktrace_entry> always_eq2;
  auto always_eq_st1 = std::basic_stacktrace<decltype(always_eq1)>(always_eq1);
  auto always_eq_st2 = std::basic_stacktrace<decltype(always_eq2)>(always_eq2);
  static_assert(noexcept(std::swap(always_eq_st1, always_eq_st2)));

  return 0;
}
