//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// <string>

// basic_string<charT,traits,Allocator>&
//   operator=(basic_string<charT,traits,Allocator>&& str); // constexpr since C++20

#include <string>
#include <cassert>

#include "test_macros.h"
#include "asan_testing.h"

TEST_CONSTEXPR_CXX20 bool test() {
  // Test that assignment from {} and {ptr, len} are allowed and are not
  // ambiguous.
  {
    std::string s = "hello world";
    s             = {};
    assert(s.empty());
    LIBCPP_ASSERT(is_string_asan_correct(s));
  }
  {
    std::string s = "hello world";
    s             = {"abc", 2};
    assert(s == "ab");
    LIBCPP_ASSERT(is_string_asan_correct(s));
  }
  {
    std::string s = "hello world";
    s             = {"It'sALongString!NoSSO!qwertyuiop", 30};
    assert(s == "It'sALongString!NoSSO!qwertyui");
    LIBCPP_ASSERT(is_string_asan_correct(s));
  }
  {
    std::string s = "Hello world! Hello world! Hello world! Hello world! Hello world!";
    s             = {"It'sALongString!NoSSO!qwertyuiop", 30};
    assert(s == "It'sALongString!NoSSO!qwertyui");
    LIBCPP_ASSERT(is_string_asan_correct(s));
  }
  {
    std::string s = "Hello world! Hello world! Hello world! Hello world! Hello world!";
    s             = {"abc", 2};
    assert(s == "ab");
    LIBCPP_ASSERT(is_string_asan_correct(s));
  }
  {
    std::string s = "Hello world! Hello world! Hello world! Hello world! Hello world!";
    s             = {"abc", 0};
    assert(s == "");
    LIBCPP_ASSERT(is_string_asan_correct(s));
  }

  return true;
}

int main(int, char**) {
  test();
#if TEST_STD_VER > 17
  static_assert(test());
#endif

  return 0;
}
