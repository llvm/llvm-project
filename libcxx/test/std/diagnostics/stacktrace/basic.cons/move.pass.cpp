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
  (19.6.4.2)

  // [stacktrace.basic.cons], creation and assignment
  basic_stacktrace(basic_stacktrace&& other) noexcept;                                    
  basic_stacktrace(basic_stacktrace&& other, const allocator_type& alloc);                
  basic_stacktrace& operator=(basic_stacktrace&& other)
    noexcept(allocator_traits<Allocator>::propagate_on_container_move_assignment::value ||
      allocator_traits<Allocator>::is_always_equal::value);                               
*/

#include <cassert>
#include <stacktrace>
#include <utility>

#include "../test_allocs.h"

void test_move_construct() {
  auto a = std::stacktrace::current();
  std::stacktrace b{a};
  assert(a == b);
}

void test_move_assign() {
  {
    using A =
        TestAlloc<std::stacktrace_entry,
                  /*_KNoExCtors=*/true,
                  /*_KNoExAlloc=*/true,
                  /*_KPropagate=*/false,
                  /*_KAlwaysEqual=*/false>;
    auto s0 = std::basic_stacktrace<A>::current();
    std::basic_stacktrace<A> s1{s0};
    std::basic_stacktrace<A> s2(std::move(s0));
    assert(s1 == s2);
    auto a1 = s1.get_allocator();
    auto a2 = s2.get_allocator();
    // Allocator should not propagate
    assert(a1 != a2);
  }
  {
    using A =
        TestAlloc<std::stacktrace_entry,
                  /*_KNoExCtors=*/true,
                  /*_KNoExAlloc=*/true,
                  /*_KPropagate=*/true,
                  /*_KAlwaysEqual=*/false>;
    auto s0 = std::basic_stacktrace<A>::current();
    std::basic_stacktrace<A> s1{s0};
    std::basic_stacktrace<A> s2(std::move(s0));
    auto a1 = s1.get_allocator();
    auto a2 = s2.get_allocator();
    // Allocator should propagate
    assert(a1 == a2);
  }
  {
    using A =
        TestAlloc<std::stacktrace_entry,
                  /*_KNoExCtors=*/true,
                  /*_KNoExAlloc=*/true,
                  /*_KPropagate=*/false,
                  /*_KAlwaysEqual=*/true>;
    auto s0 = std::basic_stacktrace<A>::current();
    std::basic_stacktrace<A> s1{s0};
    std::basic_stacktrace<A> s2(std::move(s0));
    auto a1 = s1.get_allocator();
    auto a2 = s2.get_allocator();
    // Allocator should propagate
    assert(a1 == a2);
  }
}

_LIBCPP_NO_TAIL_CALLS
int main(int, char**) {
  test_move_construct();
  test_move_assign();
  return 0;
}
