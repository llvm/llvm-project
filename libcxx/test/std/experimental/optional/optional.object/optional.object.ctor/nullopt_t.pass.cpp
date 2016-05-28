//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03, c++11
// <optional>

// constexpr optional(nullopt_t) noexcept;

#include <experimental/optional>
#include <type_traits>
#include <cassert>

using std::experimental::optional;
using std::experimental::nullopt_t;
using std::experimental::nullopt;

template <class Opt>
void
test_constexpr()
{
    static_assert(noexcept(Opt(nullopt)), "");
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
    static_assert(noexcept(Opt(nullopt)), "");
    Opt opt(nullopt);
    assert(static_cast<bool>(opt) == false);

    struct test_constexpr_ctor
        : public Opt
    {
        constexpr test_constexpr_ctor() {}
    };
}

struct X
{
    X();
};

int main()
{
    test_constexpr<optional<int>>();
    test_constexpr<optional<int*>>();
    test<optional<X>>();
}
