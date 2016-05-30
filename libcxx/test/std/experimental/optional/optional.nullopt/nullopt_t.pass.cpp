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

// struct nullopt_t{see below};
// constexpr nullopt_t nullopt(unspecified);

#include <experimental/optional>
#include <type_traits>

using std::experimental::optional;
using std::experimental::nullopt_t;
using std::experimental::nullopt;

constexpr
int
test(const nullopt_t&)
{
    return 3;
}

int main()
{
    static_assert((std::is_class<nullopt_t>::value), "");
    static_assert((std::is_empty<nullopt_t>::value), "");
    static_assert((std::is_literal_type<nullopt_t>::value), "");
    static_assert((!std::is_default_constructible<nullopt_t>::value), "");
    
    static_assert(test(nullopt) == 3, "");
}
