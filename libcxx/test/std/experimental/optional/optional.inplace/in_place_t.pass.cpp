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

// struct in_place_t{};
// constexpr in_place_t in_place{};

#include <experimental/optional>
#include <type_traits>

using std::experimental::optional;
using std::experimental::in_place_t;
using std::experimental::in_place;

constexpr
int
test(const in_place_t&)
{
    return 3;
}

int main()
{
    static_assert((std::is_class<in_place_t>::value), "");
    static_assert((std::is_empty<in_place_t>::value), "");

    static_assert(test(in_place) == 3, "");
}
