//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03, c++11, c++14

// <optional>

// optional(optional<T>&& rhs);

#include <optional>
#include <string>
#include <type_traits>
#include <utility>

using std::optional;

struct X {};

struct Y
{
    Y() = default;
    Y(Y&&) {}
};

struct Z
{
    Z() = default;
    Z(Z&&) = delete;
    Z(const Z&) = delete;
    Z& operator=(Z&&) = delete;
    Z& operator=(const Z&) = delete;
};

int main()
{
    {
        using T = int;
        static_assert((std::is_trivially_copy_constructible<optional<T>>::value), "");
        constexpr optional<T> opt;
        constexpr optional<T> opt2 = std::move(opt);
        (void)opt2;
    }
    {
        using T = X;
        static_assert((std::is_trivially_copy_constructible<optional<T>>::value), "");
        constexpr optional<T> opt;
        constexpr optional<T> opt2 = std::move(opt);
        (void)opt2;
    }
    static_assert(!(std::is_trivially_move_constructible<optional<Y>>::value), "");
    static_assert(!(std::is_trivially_move_constructible<optional<std::string>>::value), "");

    static_assert(!(std::is_move_constructible<optional<Z>>::value), "");
}
