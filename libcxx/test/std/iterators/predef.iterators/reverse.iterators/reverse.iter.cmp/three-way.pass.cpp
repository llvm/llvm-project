//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-no-concepts

// <iterator>

// reverse_iterator

// template<class Iterator1, three_way_comparable_with<Iterator1> Iterator2>
//   constexpr compare_three_way_result_t<Iterator1, Iterator2>
//    operator<=>(const reverse_iterator<Iterator1>& x,
//                const reverse_iterator<Iterator2>& y);

#include <iterator>
#include <limits>
#include <cassert>

#include "test_macros.h"
#include "test_iterators.h"

template <class ItL, class ItR, class Ord>
constexpr void test(ItL l, ItR r, Ord x) {
    const std::reverse_iterator<ItL> l1(l);
    const std::reverse_iterator<ItR> r1(r);
    ASSERT_SAME_TYPE(decltype(l1 <=> r1), Ord);
    assert((l1 <=> r1) == x);
}

struct Iter {
    using iterator_category = std::bidirectional_iterator_tag;
    using pointer = char*;
    using reference = char&;
    using value_type = char;
    using difference_type = double;

    constexpr Iter(double value): m_value(value) {}
    double m_value;

    reference operator*() const;
private:
    friend constexpr bool operator==(const Iter& l, const Iter& r) = default;
    friend constexpr std::partial_ordering operator<=>(const Iter& l, const Iter& r) = default;
};

struct ConstIter {
    using iterator_category = std::bidirectional_iterator_tag;
    using pointer = const char*;
    using reference = const char&;
    using value_type = const char;
    using difference_type = double;

    constexpr ConstIter(double value): m_value(value) {}
    constexpr ConstIter(Iter it): m_value(it.m_value) {}
    double m_value;

    reference operator*() const;
private:
    friend constexpr bool operator==(const ConstIter& l, const ConstIter& r) = default;
    friend constexpr std::partial_ordering operator<=>(const ConstIter& l, const ConstIter& r) = default;
};

constexpr bool tests() {
    char s[] = "1234567890";
    test(three_way_contiguous_iterator<const char*>(s),
         three_way_contiguous_iterator<const char*>(s),
         std::strong_ordering::equal);
    test(three_way_contiguous_iterator<const char*>(s),
         three_way_contiguous_iterator<const char*>(s+1),
         std::strong_ordering::greater);
    test(three_way_contiguous_iterator<const char*>(s+1),
         three_way_contiguous_iterator<const char*>(s),
         std::strong_ordering::less);

    test(s, s, std::strong_ordering::equal);
    test(s, s+1, std::strong_ordering::greater);
    test(s+1, s, std::strong_ordering::less);

    const char* cs = s;
    test(cs, s, std::strong_ordering::equal);
    test(cs, s+1, std::strong_ordering::greater);
    test(cs+1, s, std::strong_ordering::less);

    constexpr double nan = std::numeric_limits<double>::quiet_NaN();
    test(Iter(0), ConstIter(nan), std::partial_ordering::unordered);
    test(Iter(nan), Iter(nan), std::partial_ordering::unordered);
    test(ConstIter(0), Iter(1), std::partial_ordering::greater);
    test(ConstIter(3), Iter(2), std::partial_ordering::less);
    test(ConstIter(7), Iter(7), std::partial_ordering::equivalent);

    return true;
}

int main(int, char**) {
    tests();
    static_assert(tests());
    return 0;
}
