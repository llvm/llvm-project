//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_DISABLE_DEPRECATION_WARNINGS

// <strstream>

// class strstream

// strstream();

#include <strstream>
#include <cassert>
#include <cstring>
#include <string>

#include "test_macros.h"

int main(int, char**)
{
    std::strstream inout;
    int i = 123;
    double d = 4.5;
    std::string s("dog");
    inout << i << ' ' << d << ' ' << s << std::ends;
    assert(inout.str() == std::string("123 4.5 dog"));
    i = 0;
    d = 0;
    s = "";
    inout >> i >> d >> s;
    assert(i == 123);
    assert(d == 4.5);
    assert(std::strcmp(s.c_str(), "dog") == 0);
    inout.freeze(false);

  return 0;
}
