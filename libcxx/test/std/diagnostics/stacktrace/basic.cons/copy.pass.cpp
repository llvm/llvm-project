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
//   basic_stacktrace(const basic_stacktrace& other);
//   basic_stacktrace(const basic_stacktrace& other, const allocator_type& alloc);
//   basic_stacktrace& operator=(const basic_stacktrace& other);

#include <cassert>
#include <stacktrace>

#include "test_allocator.h"
#include "test_macros.h"

// Self-assignment must be well-defined (leaves the object unchanged).
TEST_CLANG_DIAGNOSTIC_IGNORED("-Wself-assign-overloaded")

int main() {
  // Copy-construction tests

  {
    auto s0 = std::stacktrace::current();
    std::stacktrace s1{s0};
    assert(s1 == s0);
  }

  {
    auto s0 = std::stacktrace::current();
    std::stacktrace s1{s0};
    assert(s1 == s0);
  }

  // Copy-construction with an explicit allocator

  {
    auto s0 = std::stacktrace::current();
    std::stacktrace s1{s0, std::allocator<std::stacktrace_entry>()};
    assert(s1 == s0);
  }

  // Copy-construction (without an explicit allocator) must apply
  // allocator_traits<Allocator>::select_on_container_copy_construction(other.get_allocator()).
  // `test_allocator` doesn't override that trait, so the default behavior applies: the new
  // object's allocator must be a copy of `other`'s allocator, not a default-constructed one.
  {
    using A = test_allocator<std::stacktrace_entry>;
    A alloc(42);
    std::basic_stacktrace<A> s0(alloc);
    std::basic_stacktrace<A> s1{s0};
    assert(s1.get_allocator().get_data() == 42);
  }

  // Copy-assignment tests

  {
    auto s0 = std::stacktrace::current();
    std::stacktrace s1{s0};
    s1 = s0;
    assert(s1 == s0);
  }

  // Self-copy-assignment must leave the object unchanged.

  {
    auto s0              = std::stacktrace::current();
    auto const s0_before = s0;
    s0                   = s0;
    assert(s0 == s0_before);
  }

  return 0;
}
