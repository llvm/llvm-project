//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <string>

// Index iterator out of bounds.

// REQUIRES: has-unix-headers, libcpp-has-abi-bounded-iterators-in-string
// UNSUPPORTED: libcpp-hardening-mode=none, c++03

#include <string>
#include <cassert>

#include "check_assertion.h"
#include "min_allocator.h"

template <class C>
void test() {
  using T = decltype(std::uint8_t() - std::uint8_t());
  C c(1, '\0');
  typename C::iterator i = c.begin();
  assert(i[0] == 0);
  TEST_LIBCPP_ASSERT_FAILURE(i[1], "__bounded_iter::operator[]: Attempt to index an iterator at or past the end");
  TEST_LIBCPP_ASSERT_FAILURE(i[-1], "__bounded_iter::operator[]: Attempt to index an iterator past the start");
}

int main(int, char**) {
  test<std::string>();
  test<std::basic_string<char, std::char_traits<char>, min_allocator<char> > >();

  return 0;
}
