//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// <charconv>

// constexpr from_chars_result from_chars(const char* first, const char* last,
//                                        Float& value, chars_format fmt = chars_format::general)

#include <charconv>
#include "test_macros.h"
#include "charconv_test_helpers.h"

template <typename T>
struct test_basics {
  void operator()() {
    std::from_chars_result r;
    T x;

    {
      char s[] = "001x";

      // the expected form of the subject sequence is a nonempty sequence of
      // decimal digits optionally containing a decimal-point character, then
      // an optional exponent part as defined in 6.4.4.3, excluding any digit
      // separators (6.4.4.2); (C23 7.24.1.5)
      r = std::from_chars(s, s + sizeof(s), x);
      assert(r.ec == std::errc{});
      assert(r.ptr == s + 3);
      assert(x == T(1.0));
    }

    {
      char s[] = "1.5e10";

      r = std::from_chars(s, s + sizeof(s), x);
      assert(r.ec == std::errc{});
      assert(r.ptr == s + 6);
      assert(x == T(1.5e10));
    }

    {
      char s[] = "20040229";

      // This number is halfway between two float values.
      r = std::from_chars(s, s + sizeof(s), x);
      assert(r.ec == std::errc{});
      assert(r.ptr == s + 8);
      assert(x == T(20040229));
    }
  }
};

bool test() {
  run<test_basics>(all_floats);

  return true;
}

int main(int, char**) {
  test();

  return 0;
}
