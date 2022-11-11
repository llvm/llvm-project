//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <string>

// Check that basic_string(basic_string&&, size_type, Allocator) and
// basic_string(basic_string&&, size_type, size_type, Allocator) inserts the container into the debug database

// REQUIRES: has-unix-headers
// UNSUPPORTED: !libcpp-has-debug-mode, c++03

#include <cassert>
#include <string>

#include "check_assertion.h"

int main(int, char**) {
  using namespace std::string_literals;

  {
    std::string s = {"Banane"s, 1};
    auto i        = s.begin();
    assert(i[0] == 'a');
    TEST_LIBCPP_ASSERT_FAILURE(i[5], "Attempted to subscript an iterator outside its valid range");
  }
  {
    std::string s = {"Banane"s, 0, 5};
    auto i        = s.begin();
    assert(i[0] == 'B');
    TEST_LIBCPP_ASSERT_FAILURE(i[5], "Attempted to subscript an iterator outside its valid range");
  }
  {
    std::string s = {"long long string so no SSO"s, 21};
    auto i        = s.begin();
    assert(i[0] == 'o');
    TEST_LIBCPP_ASSERT_FAILURE(i[5], "Attempted to subscript an iterator outside its valid range");
  }
  {
    std::string s = {"long long string so no SSO"s, 0, 5};
    auto i        = s.begin();
    assert(i[0] == 'l');
    TEST_LIBCPP_ASSERT_FAILURE(i[5], "Attempted to subscript an iterator outside its valid range");
  }
}
