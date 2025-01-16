//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23

/*
  (19.6.4.6) Non-member functions

  template<class Allocator>
    void swap(basic_stacktrace<Allocator>& a, basic_stacktrace<Allocator>& b)
      noexcept(noexcept(a.swap(b)));
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

  std::swap(a, b);
  assert(a == current);
  assert(b == empty);

  return 0;
}
