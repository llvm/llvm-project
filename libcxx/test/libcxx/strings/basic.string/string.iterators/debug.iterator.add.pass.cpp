//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <string>

// Add to iterator out of bounds.

// REQUIRES: has-unix-headers
// UNSUPPORTED: !libcpp-has-legacy-debug-mode, c++03

#include <string>
#include <cassert>

#include "check_assertion.h"
#include "min_allocator.h"

template <class C>
void test() {
  C c(1, '\0');
  typename C::iterator i = c.begin();
  i += 1;
  assert(i == c.end());
  i = c.begin();
  TEST_LIBCPP_ASSERT_FAILURE(i += 2, "Attempted to add/subtract an iterator outside its valid range");
}

int main(int, char**) {
  test<std::string>();
  test<std::basic_string<char, std::char_traits<char>, min_allocator<char> > >();

  return 0;
}
