//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <string>

// iterator insert(const_iterator p, size_type n, charT c);

// REQUIRES: has-unix-headers
// UNSUPPORTED: !libcpp-has-legacy-debug-mode, c++03

#include <string>

#include "check_assertion.h"
#include "test_allocator.h"

template <class S>
void test() {
  S s;
  S s2;
  TEST_LIBCPP_ASSERT_FAILURE(s.insert(s2.begin(), 1, 'a'),
                             "string::insert(iterator, n, value) called with an iterator not referring to this string");
}

int main(int, char**) {
  test<std::string>();
  test<std::basic_string<char, std::char_traits<char>, fancy_pointer_allocator<char> > >();

  return 0;
}
