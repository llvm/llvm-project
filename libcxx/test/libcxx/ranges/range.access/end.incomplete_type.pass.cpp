//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-no-concepts
// UNSUPPORTED: libcpp-has-no-incomplete-ranges

// Test the libc++-specific behavior that we handle the IFNDR case for ranges::end
// by being SFINAE-friendly.

#include <ranges>
#include <cassert>
#include <type_traits>

struct Incomplete;

constexpr bool test()
{
    {
    extern Incomplete bounded[10];
    assert((!std::is_invocable_v<decltype(std::ranges::end), decltype((bounded))>));
    assert((!std::is_invocable_v<decltype(std::ranges::cend), decltype((bounded))>));
    assert((!std::is_invocable_v<decltype(std::ranges::end), decltype(std::as_const(bounded))>));
    assert((!std::is_invocable_v<decltype(std::ranges::cend), decltype(std::as_const(bounded))>));
    }
    {
    extern Incomplete unbounded[];
    assert((!std::is_invocable_v<decltype(std::ranges::end), decltype((unbounded))>));
    assert((!std::is_invocable_v<decltype(std::ranges::cend), decltype((unbounded))>));
    assert((!std::is_invocable_v<decltype(std::ranges::end), decltype(std::as_const(unbounded))>));
    assert((!std::is_invocable_v<decltype(std::ranges::cend), decltype(std::as_const(unbounded))>));
    }

    return true;
}

int main(int, char**)
{
    test();
    static_assert(test());
}
