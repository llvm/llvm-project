//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// <compare>

// constexpr bool is_eq  (partial_ordering cmp) noexcept { return cmp == 0; }
// constexpr bool is_neq (partial_ordering cmp) noexcept { return cmp != 0; }
// constexpr bool is_lt  (partial_ordering cmp) noexcept { return cmp < 0; }
// constexpr bool is_lteq(partial_ordering cmp) noexcept { return cmp <= 0; }
// constexpr bool is_gt  (partial_ordering cmp) noexcept { return cmp > 0; }
// constexpr bool is_gteq(partial_ordering cmp) noexcept { return cmp >= 0; }

#include <compare>
#include <cassert>

#include "test_macros.h"

constexpr bool test()
{
    assert(!std::is_eq(std::strong_ordering::less));
    assert( std::is_eq(std::strong_ordering::equal));
    assert(!std::is_eq(std::strong_ordering::greater));
    assert(!std::is_eq(std::weak_ordering::less));
    assert( std::is_eq(std::weak_ordering::equivalent));
    assert(!std::is_eq(std::weak_ordering::greater));
    assert(!std::is_eq(std::partial_ordering::less));
    assert( std::is_eq(std::partial_ordering::equivalent));
    assert(!std::is_eq(std::partial_ordering::greater));
    assert(!std::is_eq(std::partial_ordering::unordered));

    assert( std::is_neq(std::strong_ordering::less));
    assert(!std::is_neq(std::strong_ordering::equal));
    assert( std::is_neq(std::strong_ordering::greater));
    assert( std::is_neq(std::weak_ordering::less));
    assert(!std::is_neq(std::weak_ordering::equivalent));
    assert( std::is_neq(std::weak_ordering::greater));
    assert( std::is_neq(std::partial_ordering::less));
    assert(!std::is_neq(std::partial_ordering::equivalent));
    assert( std::is_neq(std::partial_ordering::greater));
    assert( std::is_neq(std::partial_ordering::unordered));

    assert( std::is_lt(std::strong_ordering::less));
    assert(!std::is_lt(std::strong_ordering::equal));
    assert(!std::is_lt(std::strong_ordering::greater));
    assert( std::is_lt(std::weak_ordering::less));
    assert(!std::is_lt(std::weak_ordering::equivalent));
    assert(!std::is_lt(std::weak_ordering::greater));
    assert( std::is_lt(std::partial_ordering::less));
    assert(!std::is_lt(std::partial_ordering::equivalent));
    assert(!std::is_lt(std::partial_ordering::greater));
    assert(!std::is_lt(std::partial_ordering::unordered));

    assert( std::is_lteq(std::strong_ordering::less));
    assert( std::is_lteq(std::strong_ordering::equal));
    assert(!std::is_lteq(std::strong_ordering::greater));
    assert( std::is_lteq(std::weak_ordering::less));
    assert( std::is_lteq(std::weak_ordering::equivalent));
    assert(!std::is_lteq(std::weak_ordering::greater));
    assert( std::is_lteq(std::partial_ordering::less));
    assert( std::is_lteq(std::partial_ordering::equivalent));
    assert(!std::is_lteq(std::partial_ordering::greater));
    assert(!std::is_lteq(std::partial_ordering::unordered));

    assert(!std::is_gt(std::strong_ordering::less));
    assert(!std::is_gt(std::strong_ordering::equal));
    assert( std::is_gt(std::strong_ordering::greater));
    assert(!std::is_gt(std::weak_ordering::less));
    assert(!std::is_gt(std::weak_ordering::equivalent));
    assert( std::is_gt(std::weak_ordering::greater));
    assert(!std::is_gt(std::partial_ordering::less));
    assert(!std::is_gt(std::partial_ordering::equivalent));
    assert( std::is_gt(std::partial_ordering::greater));
    assert(!std::is_gt(std::partial_ordering::unordered));

    assert(!std::is_gteq(std::strong_ordering::less));
    assert( std::is_gteq(std::strong_ordering::equal));
    assert( std::is_gteq(std::strong_ordering::greater));
    assert(!std::is_gteq(std::weak_ordering::less));
    assert( std::is_gteq(std::weak_ordering::equivalent));
    assert( std::is_gteq(std::weak_ordering::greater));
    assert(!std::is_gteq(std::partial_ordering::less));
    assert( std::is_gteq(std::partial_ordering::equivalent));
    assert( std::is_gteq(std::partial_ordering::greater));
    assert(!std::is_gteq(std::partial_ordering::unordered));

    // Test noexceptness.
    ASSERT_NOEXCEPT(std::is_eq(std::partial_ordering::less));
    ASSERT_NOEXCEPT(std::is_neq(std::partial_ordering::less));
    ASSERT_NOEXCEPT(std::is_lt(std::partial_ordering::less));
    ASSERT_NOEXCEPT(std::is_lteq(std::partial_ordering::less));
    ASSERT_NOEXCEPT(std::is_gt(std::partial_ordering::less));
    ASSERT_NOEXCEPT(std::is_gteq(std::partial_ordering::less));

    return true;
}

int main(int, char**) {
    test();
    static_assert(test());

    return 0;
}
