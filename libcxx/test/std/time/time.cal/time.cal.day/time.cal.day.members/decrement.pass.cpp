//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++03, c++11, c++14, c++17

// <chrono>
// class day;

//  constexpr day& operator--() noexcept;
//  constexpr day operator--(int) noexcept;


#include <chrono>
#include <type_traits>
#include <cassert>

#include "test_macros.h"

template <typename D>
constexpr bool testConstexpr()
{
    D d1{10};
    if (static_cast<unsigned>(--d1) != 9) return false;
    if (static_cast<unsigned>(d1--) != 9) return false;
    if (static_cast<unsigned>(d1)   != 8) return false;
    return true;
}

int main(int, char**)
{
    using day = std::chrono::day;
    ASSERT_NOEXCEPT(--(std::declval<day&>())  );
    ASSERT_NOEXCEPT(  (std::declval<day&>())--);

    ASSERT_SAME_TYPE(day , decltype(  std::declval<day&>()--));
    ASSERT_SAME_TYPE(day&, decltype(--std::declval<day&>()  ));

    static_assert(testConstexpr<day>(), "");

    for (unsigned i = 10; i <= 20; ++i)
    {
        day d(i);
        assert(static_cast<unsigned>(--d) == i - 1);
        assert(static_cast<unsigned>(d--) == i - 1);
        assert(static_cast<unsigned>(d)   == i - 2);
    }

  return 0;
}
