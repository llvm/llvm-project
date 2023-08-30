//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: has-unix-headers
// UNSUPPORTED: c++03
// UNSUPPORTED: !libcpp-has-debug-mode && !libcpp-has-assertions
// XFAIL: availability-verbose_abort-missing

// <filesystem>

// class path

#include "filesystem_include.h"
#include <iterator>
#include <type_traits>
#include <cassert>

#include "check_assertion.h"

int main(int, char**) {
  // Test incrementing/decrementing a singular iterator
  {
    fs::path::iterator singular;
    TEST_LIBCPP_ASSERT_FAILURE(++singular, "attempting to increment a singular iterator");
    TEST_LIBCPP_ASSERT_FAILURE(--singular, "attempting to decrement a singular iterator");
  }

  // Test incrementing the end iterator
  {
    fs::path p("foo/bar");
    auto it = p.begin();
    TEST_LIBCPP_ASSERT_FAILURE(--it, "attempting to decrement the begin iterator");
  }

  // Test incrementing the end iterator
  {
    fs::path p("foo/bar");
    auto it = p.end();
    TEST_LIBCPP_ASSERT_FAILURE(++it, "attempting to increment the end iterator");
  }

  return 0;
}
