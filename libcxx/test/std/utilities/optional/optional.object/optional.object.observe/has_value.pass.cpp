//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14
// <optional>

// constexpr bool optional<T>::has_value() const noexcept;

#include <cassert>
#include <optional>

#include "test_macros.h"

int main(int, char**)
{
    using std::optional;
    {
        const optional<int> opt; ((void)opt);
        ASSERT_NOEXCEPT(opt.has_value());
        ASSERT_SAME_TYPE(decltype(opt.has_value()), bool);
    }
    {
        constexpr optional<int> opt;
        static_assert(!opt.has_value(), "");
    }
    {
        constexpr optional<int> opt(0);
        static_assert(opt.has_value(), "");
    }
#if TEST_STD_VER >= 26
    {
      static constexpr int i = 0;
      constexpr optional<const int&> opt{i};
      ASSERT_NOEXCEPT(opt.has_value());
      static_assert(opt.has_value());
    }
    {
      constexpr optional<const int&> opt{};
      static_assert(!opt.has_value());
    }
#endif

    return 0;
}
