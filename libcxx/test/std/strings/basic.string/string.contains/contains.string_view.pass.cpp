//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <string>

// constexpr bool contains(basic_string_view x) const noexcept;

#include <string>
#include <cassert>

#include "test_macros.h"

constexpr bool test()
{
    using S = std::string;
    using SV = std::string_view;

    const char* s = "abcde";
    S s0;
    S s1 {s + 1, 1};
    S s3 {s + 1, 3};
    S s5 {s    , 5};
    S sNot {"xyz", 3};

    SV sv0;
    SV sv1 {s + 1, 1};
    SV sv2 {s + 1, 2};
    SV sv3 {s + 1, 3};
    SV sv4 {s + 1, 4};
    SV sv5 {s    , 5};
    SV svNot  {"xyz", 3};
    SV svNot2 {"bd" , 2};
    SV svNot3 {"dcb", 3};

    ASSERT_NOEXCEPT(s0.contains(sv0));

    assert( s0.contains(sv0));
    assert(!s0.contains(sv1));

    assert( s1.contains(sv0));
    assert( s1.contains(sv1));
    assert(!s1.contains(sv2));
    assert(!s1.contains(sv3));
    assert(!s1.contains(sv4));
    assert(!s1.contains(sv5));
    assert(!s1.contains(svNot));
    assert(!s1.contains(svNot2));
    assert(!s1.contains(svNot3));

    assert( s3.contains(sv0));
    assert( s3.contains(sv1));
    assert( s3.contains(sv2));
    assert( s3.contains(sv3));
    assert(!s3.contains(sv4));
    assert(!s3.contains(sv5));
    assert(!s3.contains(svNot));
    assert(!s3.contains(svNot2));
    assert(!s3.contains(svNot3));

    assert( s5.contains(sv0));
    assert( s5.contains(sv1));
    assert( s5.contains(sv2));
    assert( s5.contains(sv3));
    assert( s5.contains(sv4));
    assert( s5.contains(sv5));
    assert(!s5.contains(svNot));
    assert(!s5.contains(svNot2));
    assert(!s5.contains(svNot3));

    assert( sNot.contains(sv0));
    assert(!sNot.contains(sv1));
    assert(!sNot.contains(sv2));
    assert(!sNot.contains(sv3));
    assert(!sNot.contains(sv4));
    assert(!sNot.contains(sv5));
    assert( sNot.contains(svNot));
    assert(!sNot.contains(svNot2));
    assert(!sNot.contains(svNot3));

    return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
