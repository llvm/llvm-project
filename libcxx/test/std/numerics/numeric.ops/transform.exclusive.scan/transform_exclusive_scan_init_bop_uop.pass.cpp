//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// <numeric>

// Became constexpr in C++20
// template<class InputIterator, class OutputIterator, class T,
//          class BinaryOperation, class UnaryOperation>
//   OutputIterator transform_exclusive_scan(InputIterator first, InputIterator last,
//                                           OutputIterator result, T init,
//                                           BinaryOperation binary_op,
//                                           UnaryOperation unary_op);

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <functional>
#include <numeric>

#include "test_macros.h"
#include "test_iterators.h"

struct add_one {
    template <typename T>
    constexpr auto operator()(T x) const noexcept {
        return static_cast<T>(x + 1);
    }
};

template <class Iter1, class BOp, class UOp, class T>
TEST_CONSTEXPR_CXX20 void
test(Iter1 first, Iter1 last, BOp bop, UOp uop, T init, const T *rFirst, const T *rLast)
{
    assert((rLast - rFirst) <= 5);  // or else increase the size of "out"
    T out[5];

    // Not in place
    T *end = std::transform_exclusive_scan(first, last, out, init, bop, uop);
    assert(std::equal(out, end, rFirst, rLast));

    // In place
    std::copy(first, last, out);
    end = std::transform_exclusive_scan(out, end, out, init, bop, uop);
    assert(std::equal(out, end, rFirst, rLast));
}

template <class Iter>
TEST_CONSTEXPR_CXX20 void
test()
{
    int ia[]           = { 1,  3,  5,    7,   9 };
    const int pResI0[] = { 0,  2,  6,   12,  20 };        // with add_one
    const int mResI0[] = { 0,  0,  0,    0,   0 };
    const int pResN0[] = { 0, -1, -4,   -9, -16 };        // with negate
    const int mResN0[] = { 0,  0,  0,    0,   0 };
    const int pResI2[] = { 2,  4,  8,   14,  22 };        // with add_one
    const int mResI2[] = { 2,  4, 16,   96, 768 };
    const int pResN2[] = { 2,  1,  -2,  -7, -14 };        // with negate
    const int mResN2[] = { 2, -2,   6, -30, 210 };
    const unsigned sa = sizeof(ia) / sizeof(ia[0]);
    static_assert(sa == sizeof(pResI0) / sizeof(pResI0[0]));       // just to be sure
    static_assert(sa == sizeof(mResI0) / sizeof(mResI0[0]));       // just to be sure
    static_assert(sa == sizeof(pResN0) / sizeof(pResN0[0]));       // just to be sure
    static_assert(sa == sizeof(mResN0) / sizeof(mResN0[0]));       // just to be sure
    static_assert(sa == sizeof(pResI2) / sizeof(pResI2[0]));       // just to be sure
    static_assert(sa == sizeof(mResI2) / sizeof(mResI2[0]));       // just to be sure
    static_assert(sa == sizeof(pResN2) / sizeof(pResN2[0]));       // just to be sure
    static_assert(sa == sizeof(mResN2) / sizeof(mResN2[0]));       // just to be sure

    for (unsigned int i = 0; i < sa; ++i ) {
        test(Iter(ia), Iter(ia + i), std::plus<>(),       add_one{},       0, pResI0, pResI0 + i);
        test(Iter(ia), Iter(ia + i), std::multiplies<>(), add_one{},       0, mResI0, mResI0 + i);
        test(Iter(ia), Iter(ia + i), std::plus<>(),       std::negate<>(), 0, pResN0, pResN0 + i);
        test(Iter(ia), Iter(ia + i), std::multiplies<>(), std::negate<>(), 0, mResN0, mResN0 + i);
        test(Iter(ia), Iter(ia + i), std::plus<>(),       add_one{},       2, pResI2, pResI2 + i);
        test(Iter(ia), Iter(ia + i), std::multiplies<>(), add_one{},       2, mResI2, mResI2 + i);
        test(Iter(ia), Iter(ia + i), std::plus<>(),       std::negate<>(), 2, pResN2, pResN2 + i);
        test(Iter(ia), Iter(ia + i), std::multiplies<>(), std::negate<>(), 2, mResN2, mResN2 + i);
    }
}

constexpr std::size_t triangle(size_t n) { return n*(n+1)/2; }

//  Basic sanity
TEST_CONSTEXPR_CXX20 void
basic_tests()
{
    {
    std::array<std::size_t, 10> v;
    std::fill(v.begin(), v.end(), 3);
    std::transform_exclusive_scan(v.begin(), v.end(), v.begin(), std::size_t{50}, std::plus<>(), add_one{});
    for (std::size_t i = 0; i < v.size(); ++i)
        assert(v[i] == 50 + i * 4);
    }

    {
    std::array<std::size_t, 10> v;
    std::iota(v.begin(), v.end(), 0);
    std::transform_exclusive_scan(v.begin(), v.end(), v.begin(), std::size_t{30}, std::plus<>(), add_one{});
    for (std::size_t i = 0; i < v.size(); ++i)
        assert(v[i] == 30 + triangle(i - 1) + i);
    }

    {
    std::array<std::size_t, 10> v;
    std::iota(v.begin(), v.end(), 1);
    std::transform_exclusive_scan(v.begin(), v.end(), v.begin(), std::size_t{40}, std::plus<>(), add_one{});
    for (std::size_t i = 0; i < v.size(); ++i)
        assert(v[i] == 40 + triangle(i) + i);
    }

    {
    std::array<std::size_t, 0> v, res;
    std::transform_exclusive_scan(v.begin(), v.end(), res.begin(), std::size_t{40}, std::plus<>(), add_one{});
    assert(res.empty());
    }

//  Make sure that the calculations are done using the init typedef
    {
    std::array<unsigned char, 10> v;
    std::iota(v.begin(), v.end(), static_cast<unsigned char>(1));
    std::array<std::size_t, 10> res;
    std::transform_exclusive_scan(v.begin(), v.end(), res.begin(), std::size_t{1}, std::multiplies<>(), add_one{});

    assert(res.size() == 10);
    std::size_t j = 1;
    assert(res[0] == 1);
    for (std::size_t i = 1; i < res.size(); ++i)
    {
        j *= i + 1;
        assert(res[i] == j);
    }
    }
}

TEST_CONSTEXPR_CXX20 bool
test()
{
    basic_tests();

//  All the iterator categories
    test<cpp17_input_iterator        <const int*> >();
    test<forward_iterator      <const int*> >();
    test<bidirectional_iterator<const int*> >();
    test<random_access_iterator<const int*> >();
    test<const int*>();
    test<      int*>();

    return true;
}

int main(int, char**)
{
    test();
#if TEST_STD_VER > 17
    static_assert(test());
#endif
    return 0;
}
