//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// This test requires the dylib support introduced in D92214.
// XFAIL: use_system_cxx_lib && target={{.+}}-apple-macosx10.{{9|10|11|12|13|14|15}}

// <format>

// class format_error;

#include <format>
#include <type_traits>
#include <cstring>
#include <string>
#include <cassert>

#include "test_macros.h"

int main(int, char**) {
  static_assert(std::is_base_of_v<std::runtime_error, std::format_error>);
  static_assert(std::is_polymorphic_v<std::format_error>);

  {
    const char* msg = "format_error message c-string";
    std::format_error e(msg);
    assert(std::strcmp(e.what(), msg) == 0);
    std::format_error e2(e);
    assert(std::strcmp(e2.what(), msg) == 0);
    e2 = e;
    assert(std::strcmp(e2.what(), msg) == 0);
  }
  {
    std::string msg("format_error message std::string");
    std::format_error e(msg);
    assert(e.what() == msg);
    std::format_error e2(e);
    assert(e2.what() == msg);
    e2 = e;
    assert(e2.what() == msg);
  }

  return 0;
}
