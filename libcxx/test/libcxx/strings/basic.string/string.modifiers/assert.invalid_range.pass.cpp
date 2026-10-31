//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: can-test-hardening-assertions-fast
// UNSUPPORTED: libcpp-assertion-semantic={{ignore|observe}}

// <string>

// Make sure insert, append and assign catch an iterator range with first > last.

#include <string>

#include "check_assertion.h"
#include "min_allocator.h"
#include "test_iterators.h"

template <class S>
void test() {
  S s("0123456789");
  const char a[] = "abcdef";

  TEST_LIBCPP_ASSERT_FAILURE(
      s.insert(s.cbegin(), a + 3, a), "string::insert(pos, first, last) called with invalid range");
  TEST_LIBCPP_ASSERT_FAILURE(
      s.insert(s.cbegin(), s.begin() + 8, s.begin() + 3), "string::insert(pos, first, last) called with invalid range");
  TEST_LIBCPP_ASSERT_FAILURE(
      s.insert(s.cbegin(), random_access_iterator<const char*>(a + 3), random_access_iterator<const char*>(a)),
      "string::insert(pos, first, last) called with invalid range");

  TEST_LIBCPP_ASSERT_FAILURE(s.append(a + 3, a), "string::append(first, last) called with invalid range");
  TEST_LIBCPP_ASSERT_FAILURE(
      s.append(s.begin() + 8, s.begin() + 3), "string::append(first, last) called with invalid range");
  TEST_LIBCPP_ASSERT_FAILURE(
      s.append(random_access_iterator<const char*>(a + 3), random_access_iterator<const char*>(a)),
      "string::append(first, last) called with invalid range");

  TEST_LIBCPP_ASSERT_FAILURE(s.assign(a + 3, a), "string::assign(first, last) called with invalid range");
  TEST_LIBCPP_ASSERT_FAILURE(
      s.assign(s.data() + 8, s.data() + 3), "string::assign(first, last) called with invalid range");
}

int main(int, char**) {
  test<std::string>();
  test<std::basic_string<char, std::char_traits<char>, min_allocator<char> > >();

  return 0;
}
