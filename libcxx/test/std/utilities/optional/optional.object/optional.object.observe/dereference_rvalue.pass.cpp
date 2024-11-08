//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14
// <optional>

// constexpr T&& optional<T>::operator*() &&;

#include <optional>
#include <type_traits>
#include <cassert>

#include "test_macros.h"

using std::optional;

struct X
{
    constexpr int test() const& {return 3;}
    int test() & {return 4;}
    constexpr int test() const&& {return 5;}
    int test() && {return 6;}
};

struct Y
{
    constexpr int test() && {return 7;}
};

constexpr int
test()
{
    optional<Y> opt{Y{}};
    return (*std::move(opt)).test();
}

int main(int, char**)
{
    {
        optional<X> opt; ((void)opt);
        ASSERT_SAME_TYPE(decltype(*std::move(opt)), X&&);
        ASSERT_NOEXCEPT(*std::move(opt));
    }
    {
        optional<X> opt(X{});
        assert((*std::move(opt)).test() == 6);
    }
    static_assert(test() == 7, "");

    return 0;
}
