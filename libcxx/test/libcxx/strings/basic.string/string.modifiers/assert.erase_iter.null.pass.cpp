//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <string>

// Call erase(const_iterator position) with end()

// REQUIRES: has-unix-headers
// UNSUPPORTED: c++03
// UNSUPPORTED: libcpp-hardening-mode=none
// XFAIL: libcpp-hardening-mode=debug && availability-verbose_abort-missing

#include <string>

#include "check_assertion.h"
#include "min_allocator.h"

template <class S>
void test() {
  S l1("123");
  typename S::const_iterator i = l1.end();
  TEST_LIBCPP_ASSERT_FAILURE(l1.erase(i), "string::erase(iterator) called with a non-dereferenceable iterator");
}

int main(int, char**) {
  test<std::string>();
  test<std::basic_string<char, std::char_traits<char>, min_allocator<char> > >();

  return 0;
}
