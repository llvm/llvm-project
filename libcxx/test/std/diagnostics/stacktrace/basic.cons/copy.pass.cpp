//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23
// XFAIL: availability-stacktrace-missing

// (19.6.4.2): [stacktrace.basic.cons], creation and assignment
// basic_stacktrace(const basic_stacktrace& other);
// basic_stacktrace(const basic_stacktrace& other, const allocator_type& alloc);
// basic_stacktrace& operator=(const basic_stacktrace& other);

#include <cassert>
#include <stacktrace>

#include "test_macros.h"
#include "../test_allocs.h"

void test_copy_construct() {
  std::stacktrace a = std::stacktrace::current();
  std::stacktrace b{a};
  assert(a == b);
}

void test_copy_assign() {
  {
    using A =
        TestAlloc<std::stacktrace_entry,
                  /*_KNoExCtors=*/true,
                  /*_KNoExAlloc=*/true,
                  /*_KPropagate=*/false,
                  /*_KAlwaysEqual=*/false>;
    std::basic_stacktrace<A> s1 = std::basic_stacktrace<A>::current();
    std::basic_stacktrace<A> s2{s1};
    assert(s1 == s2);
    A a1 = s1.get_allocator();
    A a2 = s2.get_allocator();
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
    std::basic_stacktrace<A> s1 = std::basic_stacktrace<A>::current();
    std::basic_stacktrace<A> s2{s1};
    assert(s1 == s2);
    A a1 = s1.get_allocator();
    A a2 = s2.get_allocator();
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
    std::basic_stacktrace<A> s1 = std::basic_stacktrace<A>::current();
    std::basic_stacktrace<A> s2{s1};
    assert(s1 == s2);
    A a1 = s1.get_allocator();
    A a2 = s2.get_allocator();
    // Allocator should propagate
    assert(a1 == a2);
  }
}

TEST_NO_TAIL_CALLS
int main(int, char**) {
  test_copy_construct();
  test_copy_assign();
  return 0;
}
