//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23

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

int main(int, char**) {
  std::stacktrace empty;
  auto current = std::stacktrace::current();

  std::stacktrace a(empty);
  std::stacktrace b(current);
  assert(a == empty);
  assert(b == current);

  a.swap(b);
  assert(a == current);
  assert(b == empty);

  // TODO(stacktrace23): should we also test swap w/ `select_on_container_swap` case

  return 0;
}
