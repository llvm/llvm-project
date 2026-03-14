//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// <utility>

// template <class T1, class T2> struct pair

// pair(const T1& x, const T2& y);

#include <type_traits>
#include <utility>

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
    explicit ExplicitNothrowT(ExplicitNothrowT const&) noexcept {}
};

struct ImplicitNothrowT {
    ImplicitNothrowT(ImplicitNothrowT const&) noexcept {}
};

int main(int, char**) {
    { // explicit noexcept test
        static_assert(!std::is_nothrow_constructible<std::pair<ExplicitT, ExplicitT>,
                                                     ExplicitT const&, ExplicitT const&>::value, "");
        static_assert(!std::is_nothrow_constructible<std::pair<ExplicitNothrowT, ExplicitT>,
                                                     ExplicitNothrowT const&, ExplicitT const&>::value, "");
        static_assert(!std::is_nothrow_constructible<std::pair<ExplicitT, ExplicitNothrowT>,
                                                     ExplicitT const&, ExplicitNothrowT const&>::value, "");
        static_assert( std::is_nothrow_constructible<std::pair<ExplicitNothrowT, ExplicitNothrowT>,
                                                     ExplicitNothrowT const&, ExplicitNothrowT const&>::value, "");
    }
    { // implicit noexcept test
        static_assert(!std::is_nothrow_constructible<std::pair<ImplicitT, ImplicitT>,
                                                     ImplicitT const&, ImplicitT const&>::value, "");
        static_assert(!std::is_nothrow_constructible<std::pair<ImplicitNothrowT, ImplicitT>,
                                                     ImplicitNothrowT const&, ImplicitT const&>::value, "");
        static_assert(!std::is_nothrow_constructible<std::pair<ImplicitT, ImplicitNothrowT>,
                                                     ImplicitT const&, ImplicitNothrowT const&>::value, "");
        static_assert( std::is_nothrow_constructible<std::pair<ImplicitNothrowT, ImplicitNothrowT>,
                                                     ImplicitNothrowT const&, ImplicitNothrowT const&>::value, "");
    }

  return 0;
}
