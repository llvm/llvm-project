//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03, c++11, c++14
// <optional>

// struct nullopt_t{see below};
// constexpr nullopt_t nullopt(unspecified);

// [optional.nullopt]/2:
//   Type nullopt_t shall not have a default constructor or an initializer-list constructor.
//   It shall not be an aggregate and shall be a literal type.
//   Constant nullopt shall be initialized with an argument of literal type.

#include <optional>
#include <type_traits>

using std::optional;
using std::nullopt_t;
using std::nullopt;

constexpr
int
test(const nullopt_t&)
{
    return 3;
}

int main()
{
    static_assert(( std::is_class<nullopt_t>::value), "");
    static_assert(( std::is_empty<nullopt_t>::value), "");
    static_assert(( std::is_literal_type<nullopt_t>::value), "");
    static_assert((!std::is_default_constructible<nullopt_t>::value), "");

    static_assert(test(nullopt) == 3, "");
}
