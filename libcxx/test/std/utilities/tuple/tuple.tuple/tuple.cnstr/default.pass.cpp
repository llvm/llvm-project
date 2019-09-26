//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <tuple>

// template <class... Types> class tuple;

// explicit(see-below) constexpr tuple();

// UNSUPPORTED: c++98, c++03

#include <tuple>
#include <string>
#include <cassert>
#include <type_traits>

#include "test_macros.h"
#include "DefaultOnly.h"

struct NoDefault {
    NoDefault() = delete;
    explicit NoDefault(int) { }
};

struct NoExceptDefault {
    NoExceptDefault() noexcept = default;
};

struct ThrowingDefault {
    ThrowingDefault() { }
};

struct IllFormedDefault {
    IllFormedDefault(int x) : value(x) {}
    template <bool Pred = false>
    constexpr IllFormedDefault() {
        static_assert(Pred,
            "The default constructor should not be instantiated");
    }
    int value;
};

int main(int, char**)
{
    {
        std::tuple<> t;
        (void)t;
    }
    {
        std::tuple<int> t;
        assert(std::get<0>(t) == 0);
    }
    {
        std::tuple<int, char*> t;
        assert(std::get<0>(t) == 0);
        assert(std::get<1>(t) == nullptr);
    }
    {
        std::tuple<int, char*, std::string> t;
        assert(std::get<0>(t) == 0);
        assert(std::get<1>(t) == nullptr);
        assert(std::get<2>(t) == "");
    }
    {
        std::tuple<int, char*, std::string, DefaultOnly> t;
        assert(std::get<0>(t) == 0);
        assert(std::get<1>(t) == nullptr);
        assert(std::get<2>(t) == "");
        assert(std::get<3>(t) == DefaultOnly());
    }
    {
        // See bug #21157.
        static_assert(!std::is_default_constructible<std::tuple<NoDefault>>(), "");
        static_assert(!std::is_default_constructible<std::tuple<DefaultOnly, NoDefault>>(), "");
        static_assert(!std::is_default_constructible<std::tuple<NoDefault, DefaultOnly, NoDefault>>(), "");
    }
    {
        static_assert(noexcept(std::tuple<NoExceptDefault>()), "");
        static_assert(noexcept(std::tuple<NoExceptDefault, NoExceptDefault>()), "");

        static_assert(!noexcept(std::tuple<ThrowingDefault, NoExceptDefault>()), "");
        static_assert(!noexcept(std::tuple<NoExceptDefault, ThrowingDefault>()), "");
        static_assert(!noexcept(std::tuple<ThrowingDefault, ThrowingDefault>()), "");
    }
    {
        constexpr std::tuple<> t;
        (void)t;
    }
    {
        constexpr std::tuple<int> t;
        assert(std::get<0>(t) == 0);
    }
    {
        constexpr std::tuple<int, char*> t;
        assert(std::get<0>(t) == 0);
        assert(std::get<1>(t) == nullptr);
    }
    {
    // Check that the SFINAE on the default constructor is not evaluated when
    // it isn't needed. If the default constructor is evaluated then this test
    // should fail to compile.
        IllFormedDefault v(0);
        std::tuple<IllFormedDefault> t(v);
    }
    {
        struct Base { };
        struct Derived : Base { protected: Derived() = default; };
        static_assert(!std::is_default_constructible<std::tuple<Derived, int> >::value, "");
    }

    return 0;
}
