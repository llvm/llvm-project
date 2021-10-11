//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-no-concepts

// <compare>
// <functional>

// compare_three_way

#include <compare>
#include <cassert>
#include <limits>
#include <type_traits>

#include "pointer_comparison_test_helper.h"

template<class T, class U>
constexpr auto test_sfinae(T t, U u)
    -> decltype(std::compare_three_way()(t, u), std::true_type{})
    { return std::true_type{}; }

constexpr auto test_sfinae(...)
    { return std::false_type{}; }

struct NotThreeWayComparable {
    std::strong_ordering operator<=>(const NotThreeWayComparable&) const;
};
ASSERT_SAME_TYPE(std::compare_three_way_result_t<NotThreeWayComparable>, std::strong_ordering);
static_assert(!std::three_way_comparable<NotThreeWayComparable>);  // it lacks operator==

struct WeaklyOrdered {
    int i;
    friend constexpr std::weak_ordering operator<=>(const WeaklyOrdered&, const WeaklyOrdered&) = default;
};

constexpr bool test()
{
    ASSERT_SAME_TYPE(decltype(std::compare_three_way()(1, 1)), std::strong_ordering);
    assert(std::compare_three_way()(1, 2) == std::strong_ordering::less);
    assert(std::compare_three_way()(1, 1) == std::strong_ordering::equal);
    assert(std::compare_three_way()(2, 1) == std::strong_ordering::greater);

    ASSERT_SAME_TYPE(decltype(std::compare_three_way()(WeaklyOrdered{1}, WeaklyOrdered{2})), std::weak_ordering);
    assert(std::compare_three_way()(WeaklyOrdered{1}, WeaklyOrdered{2}) == std::weak_ordering::less);
    assert(std::compare_three_way()(WeaklyOrdered{1}, WeaklyOrdered{1}) == std::weak_ordering::equivalent);
    assert(std::compare_three_way()(WeaklyOrdered{2}, WeaklyOrdered{1}) == std::weak_ordering::greater);

    ASSERT_SAME_TYPE(decltype(std::compare_three_way()(1.0, 1.0)), std::partial_ordering);
    double nan = std::numeric_limits<double>::quiet_NaN();
    assert(std::compare_three_way()(1.0, 2.0) == std::partial_ordering::less);
    assert(std::compare_three_way()(1.0, 1.0) == std::partial_ordering::equivalent);
    assert(std::compare_three_way()(2.0, 1.0) == std::partial_ordering::greater);
    assert(std::compare_three_way()(nan, nan) == std::partial_ordering::unordered);

    // Try heterogeneous comparison.
    ASSERT_SAME_TYPE(decltype(std::compare_three_way()(42.0, 42)), std::partial_ordering);
    assert(std::compare_three_way()(42.0, 42) == std::partial_ordering::equivalent);
    ASSERT_SAME_TYPE(decltype(std::compare_three_way()(42, 42.0)), std::partial_ordering);
    assert(std::compare_three_way()(42, 42.0) == std::partial_ordering::equivalent);

    return true;
}

int main(int, char**)
{
    test();
    static_assert(test());

    do_pointer_comparison_test(std::compare_three_way());

    static_assert(test_sfinae(1, 2));
    static_assert(!test_sfinae(1, nullptr));
    static_assert(!test_sfinae(NotThreeWayComparable(), NotThreeWayComparable()));

    return 0;
}
