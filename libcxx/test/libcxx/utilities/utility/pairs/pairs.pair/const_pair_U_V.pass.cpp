//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03

// <utility>

// template <class T1, class T2> struct pair

// template <class U, class V> EXPLICIT constexpr pair(const pair<U, V>& p);

#include <utility>

#include "test_macros.h"


struct ExplicitT {
    constexpr explicit ExplicitT(int x) : value(x) {}
    constexpr explicit ExplicitT(ExplicitT const& o) : value(o.value) {}
    int value;
};

struct ImplicitT {
    constexpr ImplicitT(int x) : value(x) {}
    constexpr ImplicitT(ImplicitT const& o) : value(o.value) {}
    int value;
};

struct ExplicitNothrowT {
    explicit ExplicitNothrowT(int x) noexcept : value(x) {}
    int value;
};

struct ImplicitNothrowT {
    ImplicitNothrowT(int x) noexcept : value(x) {}
    int value;
};

int main(int, char**) {
    { // explicit noexcept test
        static_assert(!std::is_nothrow_constructible<std::pair<ExplicitT, ExplicitT>,
                                                     std::pair<int, int> const&>::value, "");
        static_assert(!std::is_nothrow_constructible<std::pair<ExplicitNothrowT, ExplicitT>,
                                                     std::pair<int, int> const&>::value, "");
        static_assert(!std::is_nothrow_constructible<std::pair<ExplicitT, ExplicitNothrowT>,
                                                     std::pair<int, int> const&>::value, "");
        static_assert( std::is_nothrow_constructible<std::pair<ExplicitNothrowT, ExplicitNothrowT>,
                                                     std::pair<int, int> const&>::value, "");
    }
    { // implicit noexcept test
        static_assert(!std::is_nothrow_constructible<std::pair<ImplicitT, ImplicitT>,
                                                     std::pair<int, int> const&>::value, "");
        static_assert(!std::is_nothrow_constructible<std::pair<ImplicitNothrowT, ImplicitT>,
                                                     std::pair<int, int> const&>::value, "");
        static_assert(!std::is_nothrow_constructible<std::pair<ImplicitT, ImplicitNothrowT>,
                                                     std::pair<int, int> const&>::value, "");
        static_assert( std::is_nothrow_constructible<std::pair<ImplicitNothrowT, ImplicitNothrowT>,
                                                     std::pair<int, int> const&>::value, "");
    }

  return 0;
}
