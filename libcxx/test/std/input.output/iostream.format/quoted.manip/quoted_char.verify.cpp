//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <iomanip>

// quoted

// UNSUPPORTED: c++03, c++11
// XFAIL: no-wide-characters

#include <iomanip>
#include <sstream>
#include <string>
#include <cassert>

#include "test_macros.h"

//  Test that mismatches between strings and wide streams are diagnosed

void round_trip ( const char *p ) {
    std::wstringstream ss;
    ss << std::quoted(p); // expected-error {{invalid operands to binary expression}}
    std::string s;
    ss >> std::quoted(s); // expected-error {{invalid operands to binary expression}}
}

int main(int, char**) {
    round_trip("Hi Mom");
}
