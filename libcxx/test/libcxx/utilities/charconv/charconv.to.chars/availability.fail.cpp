//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03
// REQUIRES: use_system_cxx_lib && target={{.+}}-apple-macosx10.{{9|10|11|12|13|14}}

// Test the availability markup on std::to_chars.

#include <charconv>

int main(int, char**)
{
    char buf[100];
    int int_value = 33;
    long long_value = 33;
    int base = 2;
    std::to_chars(buf, buf + 100, int_value); // expected-error{{is unavailable: introduced in}}
    std::to_chars(buf, buf + 100, int_value, base); // expected-error{{is unavailable: introduced in}}

    std::to_chars(buf, buf + 100, long_value); // expected-error{{is unavailable: introduced in}}
    std::to_chars(buf, buf + 100, long_value, base); // expected-error{{is unavailable: introduced in}}
}
