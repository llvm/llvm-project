//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// ADDITIONAL_COMPILE_FLAGS: -fno-exceptions

// UNSUPPORTED: c++03

// <vector>

// Test that vector always moves elements when exceptions are disabled.
// vector is allowed to move or copy elements while resizing, so long as
// it still provides the strong exception safety guarantee.

#include <vector>
#include <cassert>

#include "test_macros.h"

#ifndef TEST_HAS_NO_EXCEPTIONS
#error exceptions should be disabled.
#endif

bool allow_moves = false;

class A {
public:
  A() {}
  A(A&&) { assert(allow_moves); }
  explicit A(int) {}
  A(A const&) { assert(false); }
};

int main(int, char**) {
  std::vector<A> v;

  // Create a vector containing some number of elements that will
  // have to be moved when it is resized.
  v.reserve(10);
  std::size_t old_cap = v.capacity();
  for (std::size_t i = 0; i < v.capacity(); ++i) {
    v.emplace_back(42);
  }
  assert(v.capacity() == old_cap);
  assert(v.size() == v.capacity());

  // The next emplace back should resize.
  allow_moves = true;
  v.emplace_back(42);

  return 0;
}
