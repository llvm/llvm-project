//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <string>

//   constexpr bool contains(charT x) const noexcept;

#include <string>
#include <cassert>

#include "test_macros.h"

void test()
{
    using S = std::string;

    S s1 {};
    S s2 {"abcde", 5};

    ASSERT_NOEXCEPT(s1.contains('e'));

    assert(!s1.contains('c'));
    assert(!s1.contains('e'));
    assert(!s1.contains('x'));
    assert( s2.contains('c'));
    assert( s2.contains('e'));
    assert(!s2.contains('x'));
}

int main(int, char**)
{
  test();
#if TEST_STD_VER > 17
  // static_assert(test());
#endif

  return 0;
}
