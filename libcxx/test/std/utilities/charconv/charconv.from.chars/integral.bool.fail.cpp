//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03, c++11
// <charconv>

// In
//
// from_chars_result from_chars(const char* first, const char* last,
//                              Integral& value, int base = 10)
//
// Integral cannot be bool.

#include <charconv>

int main(int, char**)
{
    using std::from_chars;
    char buf[] = "01001";
    bool lv;

    from_chars(buf, buf + sizeof(buf), lv);      // expected-error {{call to deleted function}}
    from_chars(buf, buf + sizeof(buf), lv, 16);  // expected-error {{call to deleted function}}

  return 0;
}
