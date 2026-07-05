//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// <charconv>

// In
//
// to_chars_result to_chars(char* first, char* last, Integral value,
//                          int base = 10)
//
// Integral cannot be bool.

#include <charconv>

int main(int, char**)
{
    using std::to_chars;
    char buf[10];
    bool lv = true;

    to_chars(buf, buf + sizeof(buf), false);   // expected-error {{call to deleted function}}
    to_chars(buf, buf + sizeof(buf), lv, 16);  // expected-error {{call to deleted function}}

  return 0;
}
