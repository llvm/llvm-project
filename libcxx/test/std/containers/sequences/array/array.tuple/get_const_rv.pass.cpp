//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <array>

// template <size_t I, class T, size_t N> const T&& get(const array<T, N>&& a);

// UNSUPPORTED: c++98, c++03

#include <array>
#include <memory>
#include <type_traits>
#include <utility>
#include <cassert>

#include "test_macros.h"

// std::array is explicitly allowed to be initialized with A a = { init-list };.
// Disable the missing braces warning for this reason.
#include "disable_missing_braces_warning.h"

int main(int, char**)
{

    {
        typedef std::unique_ptr<double> T;
        typedef std::array<T, 1> C;
        const C c = {std::unique_ptr<double>(new double(3.5))};
        static_assert(std::is_same<const T&&, decltype(std::get<0>(std::move(c)))>::value, "");
        static_assert(noexcept(std::get<0>(std::move(c))), "");
        const T&& t = std::get<0>(std::move(c));
        assert(*t == 3.5);
    }

#if TEST_STD_VER >= 14
    {
        typedef double T;
        typedef std::array<T, 3> C;
        constexpr const C c = {1, 2, 3.5};
        static_assert(std::get<0>(std::move(c)) == 1, "");
        static_assert(std::get<1>(std::move(c)) == 2, "");
        static_assert(std::get<2>(std::move(c)) == 3.5, "");
    }
#endif

    return 0;
}
