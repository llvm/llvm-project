//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <string_view>

//   constexpr bool contains(charT x) const noexcept;

#include <string_view>
#include <cassert>

#include "test_macros.h"

constexpr bool test()
{
    using SV = std::string_view;

    SV sv1 {};
    SV sv2 {"abcde", 5};

    ASSERT_NOEXCEPT(sv1.contains('e'));

    assert(!sv1.contains('c'));
    assert(!sv1.contains('e'));
    assert(!sv1.contains('x'));
    assert( sv2.contains('c'));
    assert( sv2.contains('e'));
    assert(!sv2.contains('x'));

    return true;
}

int main(int, char**)
{
    test();
    static_assert(test());

    return 0;
}
