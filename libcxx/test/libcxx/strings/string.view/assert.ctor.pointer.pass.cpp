//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: has-unix-headers
// UNSUPPORTED: c++03, c++11, c++14
// REQUIRES: libcpp-hardening-mode={{extensive|debug}}
// XFAIL: availability-verbose_abort-missing

// Construct a string_view from a null pointer
// constexpr basic_string_view( const CharT* s );

#include <string_view>

#include "check_assertion.h"

int main(int, char**) {
  TEST_LIBCPP_ASSERT_FAILURE(
      std::string_view((char const*)NULL), "null pointer passed to non-null argument of char_traits<...>::length");
  TEST_LIBCPP_ASSERT_FAILURE(
      std::string_view((char const*)nullptr), "null pointer passed to non-null argument of char_traits<...>::length");
  TEST_LIBCPP_ASSERT_FAILURE(
      std::string_view((char const*)0), "null pointer passed to non-null argument of char_traits<...>::length");
  {
    std::string_view v;
    TEST_LIBCPP_ASSERT_FAILURE(
        v == (char const*)nullptr, "null pointer passed to non-null argument of char_traits<...>::length");
    TEST_LIBCPP_ASSERT_FAILURE(
        v == (char const*)NULL, "null pointer passed to non-null argument of char_traits<...>::length");
    TEST_LIBCPP_ASSERT_FAILURE(
        (char const*)nullptr == v, "null pointer passed to non-null argument of char_traits<...>::length");
    TEST_LIBCPP_ASSERT_FAILURE(
        (char const*)NULL == v, "null pointer passed to non-null argument of char_traits<...>::length");
  }

  return 0;
}
