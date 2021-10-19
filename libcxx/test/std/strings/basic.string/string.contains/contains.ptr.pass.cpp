//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <string>

//   constexpr bool contains(const CharT *x) const;

#include <string>
#include <cassert>

#include "test_macros.h"

void test()
{
    using S = std::string;

    const char* s = "abcde";
    S s0;
    S s1 {s + 4, 1};
    S s3 {s + 2, 3};
    S sNot {"xyz", 3};

    assert(s0.contains(""));
    assert(!s0.contains("e"));

    assert( s1.contains(""));
    assert(!s1.contains("d"));
    assert( s1.contains("e"));
    assert(!s1.contains("de"));
    assert(!s1.contains("cd"));
    assert(!s1.contains("cde"));
    assert(!s1.contains("bcde"));
    assert(!s1.contains("abcde"));
    assert(!s1.contains("xyz"));

    assert( s3.contains(""));
    assert( s3.contains("d"));
    assert( s3.contains("e"));
    assert( s3.contains("de"));
    assert( s3.contains("cd"));
    assert(!s3.contains("ce"));
    assert( s3.contains("cde"));
    assert(!s3.contains("edc"));
    assert(!s3.contains("bcde"));
    assert(!s3.contains("abcde"));
    assert(!s3.contains("xyz"));

    assert( sNot.contains(""));
    assert(!sNot.contains("d"));
    assert(!sNot.contains("e"));
    assert(!sNot.contains("de"));
    assert(!sNot.contains("cd"));
    assert(!sNot.contains("cde"));
    assert(!sNot.contains("bcde"));
    assert(!sNot.contains("abcde"));
    assert( sNot.contains("xyz"));
    assert(!sNot.contains("zyx"));
}

int main(int, char**)
{
    test();

    return 0;
}
