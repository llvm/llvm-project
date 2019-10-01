//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++98, c++03, c++11, c++14, c++17

// <chrono>
// class month;

//  constexpr month& operator--() noexcept;
//  constexpr month operator--(int) noexcept;


#include <chrono>
#include <type_traits>
#include <cassert>

#include "test_macros.h"

template <typename M>
constexpr bool testConstexpr()
{
    M m1{10};
    if (static_cast<unsigned>(--m1) != 9) return false;
    if (static_cast<unsigned>(m1--) != 9) return false;
    if (static_cast<unsigned>(m1)   != 8) return false;
    return true;
}

int main(int, char**)
{
    using month = std::chrono::month;

    ASSERT_NOEXCEPT(--(std::declval<month&>())  );
    ASSERT_NOEXCEPT(  (std::declval<month&>())--);

    ASSERT_SAME_TYPE(month , decltype(  std::declval<month&>()--));
    ASSERT_SAME_TYPE(month&, decltype(--std::declval<month&>()  ));

    static_assert(testConstexpr<month>(), "");

    for (unsigned i = 10; i <= 20; ++i)
    {
        month month(i);
        assert(static_cast<unsigned>(--month) == i - 1);
        assert(static_cast<unsigned>(month--) == i - 1);
        assert(static_cast<unsigned>(month)   == i - 2);
    }

  return 0;
}
