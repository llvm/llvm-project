//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03, c++11, c++14
// <optional>

// constexpr optional(nullopt_t) noexcept;

#include <optional>
#include <type_traits>
#include <cassert>

#include "archetypes.hpp"

#include "test_macros.h"

using std::optional;
using std::nullopt_t;
using std::nullopt;

template <class Opt>
void
test_constexpr()
{
    static_assert(std::is_nothrow_constructible<Opt, nullopt_t&>::value, "");
    static_assert(std::is_trivially_destructible<Opt>::value, "");
    static_assert(std::is_trivially_destructible<typename Opt::value_type>::value, "");

    constexpr Opt opt(nullopt);
    static_assert(static_cast<bool>(opt) == false, "");

    struct test_constexpr_ctor
        : public Opt
    {
        constexpr test_constexpr_ctor() {}
    };
}

template <class Opt>
void
test()
{
    static_assert(std::is_nothrow_constructible<Opt, nullopt_t&>::value, "");
    static_assert(!std::is_trivially_destructible<Opt>::value, "");
    static_assert(!std::is_trivially_destructible<typename Opt::value_type>::value, "");
    {
    Opt opt(nullopt);
    assert(static_cast<bool>(opt) == false);
    }
    {
    const Opt opt(nullopt);
    assert(static_cast<bool>(opt) == false);
    }
    struct test_constexpr_ctor
        : public Opt
    {
        constexpr test_constexpr_ctor() {}
    };
}

int main(int, char**)
{
    test_constexpr<optional<int>>();
    test_constexpr<optional<int*>>();
    test_constexpr<optional<ImplicitTypes::NoCtors>>();
    test_constexpr<optional<NonTrivialTypes::NoCtors>>();
    test_constexpr<optional<NonConstexprTypes::NoCtors>>();
    test<optional<NonLiteralTypes::NoCtors>>();

  return 0;
}
