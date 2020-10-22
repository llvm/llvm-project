
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// <filesystem>

// class path

// operator string_type() const;

#include "filesystem_include.h"
#include <type_traits>
#include <cassert>

#include "test_macros.h"
#include "filesystem_test_helper.h"


int main(int, char**)
{
  using namespace fs;
  using string_type = path::string_type;
  const char* const value = "hello world";
  std::string value_str(value);
  fs::path::string_type pathstr_value(value_str.begin(), value_str.end());
  { // Check signature
    path p(value);
    static_assert(std::is_convertible<path, string_type>::value, "");
    static_assert(std::is_constructible<string_type, path>::value, "");
    ASSERT_SAME_TYPE(string_type, decltype(p.operator string_type()));
    ASSERT_NOT_NOEXCEPT(p.operator string_type());
  }
  {
    path p(value);
    assert(p.native() == pathstr_value);
    string_type s = p;
    assert(s == pathstr_value);
    assert(p == pathstr_value);
  }

  return 0;
}
