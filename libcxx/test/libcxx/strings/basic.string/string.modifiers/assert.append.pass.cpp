//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <string>

// basic_string& append(const value_type* s, size_type n);
// basic_string& append(const value_type* s);
// basic_string& append(const value_type* s, size_type pos, size_type n);

// REQUIRES: has-unix-headers
// UNSUPPORTED: c++03
// UNSUPPORTED: libcpp-hardening-mode={{none|fast}}
// UNSUPPORTED: libcpp-assertion-semantic={{ignore|observe}}
// XFAIL: libcpp-hardening-mode=debug && availability-verbose_abort-missing

#include <string>

#include "check_assertion.h"
#include "min_allocator.h"

template <class S>
void test() {
  S l1("123");
  const char* np = nullptr;
  TEST_LIBCPP_ASSERT_FAILURE(l1.append(np, 1), "string::append received nullptr");
  TEST_LIBCPP_ASSERT_FAILURE(l1.append(np), "string::append received nullptr");
  TEST_LIBCPP_ASSERT_FAILURE(l1.append(np, 0, 1), "string::append received nullptr");
}

int main(int, char**) {
  test<std::string>();
  test<std::basic_string<char, std::char_traits<char>, min_allocator<char> > >();

  return 0;
}
