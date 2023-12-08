//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// class std::ranges::subrange;

#include <cassert>
#include <ranges>
#include <utility>

#include "test_macros.h"

constexpr void test_sized_subrange()
{
    int a[4] = {1,2,3,4};
    auto r = std::ranges::subrange(a, a+4);
    assert(std::ranges::sized_range<decltype(r)>);
    {
        auto [first, last] = r;
        assert(first == a);
        assert(last == a+4);
    }
    {
        auto [first, last] = std::move(r);
        assert(first == a);
        assert(last == a+4);
    }
    {
        auto [first, last] = std::as_const(r);
        assert(first == a);
        assert(last == a+4);
    }
    {
        auto [first, last] = std::move(std::as_const(r));
        assert(first == a);
        assert(last == a+4);
    }
}

constexpr void test_unsized_subrange()
{
    int a[4] = {1,2,3,4};
    auto r = std::ranges::subrange(a, std::unreachable_sentinel);
    assert(!std::ranges::sized_range<decltype(r)>);
    {
        auto [first, last] = r;
        assert(first == a);
        ASSERT_SAME_TYPE(decltype(last), std::unreachable_sentinel_t);
    }
    {
        auto [first, last] = std::move(r);
        assert(first == a);
        ASSERT_SAME_TYPE(decltype(last), std::unreachable_sentinel_t);
    }
    {
        auto [first, last] = std::as_const(r);
        assert(first == a);
        ASSERT_SAME_TYPE(decltype(last), std::unreachable_sentinel_t);
    }
    {
        auto [first, last] = std::move(std::as_const(r));
        assert(first == a);
        ASSERT_SAME_TYPE(decltype(last), std::unreachable_sentinel_t);
    }
}

constexpr void test_copies_not_originals()
{
    int a[4] = {1,2,3,4};
    {
        auto r = std::ranges::subrange(a, a+4);
        auto&& [first, last] = r;
        ASSERT_SAME_TYPE(decltype(first), int*);
        ASSERT_SAME_TYPE(decltype(last), int*);
        first = a+2;
        last = a+2;
        assert(r.begin() == a);
        assert(r.end() == a+4);
    }
    {
        const auto r = std::ranges::subrange(a, a+4);
        auto&& [first, last] = r;
        ASSERT_SAME_TYPE(decltype(first), int*);
        ASSERT_SAME_TYPE(decltype(last), int*);
        first = a+2;
        last = a+2;
        assert(r.begin() == a);
        assert(r.end() == a+4);
    }
}

constexpr bool test()
{
    test_sized_subrange();
    test_unsized_subrange();
    test_copies_not_originals();
    return true;
}

int main(int, char**)
{
    test();
    static_assert(test());

    return 0;
}
