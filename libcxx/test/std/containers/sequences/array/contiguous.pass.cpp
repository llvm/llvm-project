//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <array>

// An array is a contiguous container

#include <array>
#include <cassert>
#include <memory>

#include "test_macros.h"

template <class Container>
TEST_CONSTEXPR_CXX14 void assert_contiguous(Container const& c)
{
    for (std::size_t i = 0; i < c.size(); ++i)
        assert(*(c.begin() + i) == *(std::addressof(*c.begin()) + i));
}

TEST_CONSTEXPR_CXX17 bool tests()
{
    assert_contiguous(std::array<double, 0>());
    assert_contiguous(std::array<double, 1>());
    assert_contiguous(std::array<double, 2>());
    assert_contiguous(std::array<double, 3>());

    assert_contiguous(std::array<char, 0>());
    assert_contiguous(std::array<char, 1>());
    assert_contiguous(std::array<char, 2>());
    assert_contiguous(std::array<char, 3>());

    return true;
}

int main(int, char**)
{
    tests();
#if TEST_STD_VER >= 17 // begin() & friends are constexpr in >= C++17 only
    static_assert(tests(), "");
#endif
    return 0;
}
