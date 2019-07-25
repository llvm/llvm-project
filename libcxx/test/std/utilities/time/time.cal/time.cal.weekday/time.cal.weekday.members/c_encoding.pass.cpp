//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++98, c++03, c++11, c++14, c++17

// <chrono>
// class weekday;

//  constexpr unsigned c_encoding() const noexcept;


#include <chrono>
#include <type_traits>
#include <cassert>

#include "test_macros.h"

template <typename WD>
constexpr bool testConstexpr()
{
    WD wd{5};
    return wd.c_encoding() == 5;
}

int main(int, char**)
{
    using weekday = std::chrono::weekday;

    ASSERT_NOEXCEPT(                    std::declval<weekday&>().c_encoding());
    ASSERT_SAME_TYPE(unsigned, decltype(std::declval<weekday&>().c_encoding()));

    static_assert(testConstexpr<weekday>(), "");

    for (unsigned i = 0; i <= 10; ++i)
    {
        weekday wd(i);
        assert(wd.c_encoding() == (i == 7 ? 0 : i));
    }

  return 0;
}
