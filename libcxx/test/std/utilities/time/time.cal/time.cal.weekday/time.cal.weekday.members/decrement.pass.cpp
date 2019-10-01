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

//  constexpr weekday& operator--() noexcept;
//  constexpr weekday operator--(int) noexcept;


#include <chrono>
#include <type_traits>
#include <cassert>

#include "test_macros.h"
#include "../../euclidian.h"

template <typename WD>
constexpr bool testConstexpr()
{
    WD wd{1};
    if (static_cast<unsigned>(--wd) != 0) return false;
    if (static_cast<unsigned>(wd--) != 0) return false;
    if (static_cast<unsigned>(wd)   != 6) return false;
    return true;
}

int main(int, char**)
{
    using weekday = std::chrono::weekday;
    ASSERT_NOEXCEPT(--(std::declval<weekday&>())  );
    ASSERT_NOEXCEPT(  (std::declval<weekday&>())--);

    ASSERT_SAME_TYPE(weekday , decltype(  std::declval<weekday&>()--));
    ASSERT_SAME_TYPE(weekday&, decltype(--std::declval<weekday&>()  ));

    static_assert(testConstexpr<weekday>(), "");

    for (unsigned i = 0; i <= 6; ++i)
    {
        weekday wd(i);
        assert((static_cast<unsigned>(--wd) == euclidian_subtraction<unsigned, 0, 6>(i, 1)));
        assert((static_cast<unsigned>(wd--) == euclidian_subtraction<unsigned, 0, 6>(i, 1)));
        assert((static_cast<unsigned>(wd)   == euclidian_subtraction<unsigned, 0, 6>(i, 2)));
    }

  return 0;
}
