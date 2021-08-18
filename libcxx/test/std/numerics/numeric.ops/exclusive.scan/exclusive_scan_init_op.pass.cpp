//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14
// UNSUPPORTED: clang-8

// <numeric>

// Became constexpr in C++20
// template<class InputIterator, class OutputIterator, class T, class BinaryOperation>
//     OutputIterator
//     exclusive_scan(InputIterator first, InputIterator last,
//                    OutputIterator result,
//                    T init, BinaryOperation binary_op); // C++17

#include <numeric>
#include <algorithm>
#include <array>
#include <cassert>
#include <functional>
#include <iterator>

#include "test_macros.h"
#include "test_iterators.h"

template <class Iter1, class T, class Op>
TEST_CONSTEXPR_CXX20 void
test(Iter1 first, Iter1 last, T init, Op op, const T *rFirst, const T *rLast)
{
    assert((rLast - rFirst) <= 5);  // or else increase the size of "out"
    T out[5];

    // Not in place
    T *end = std::exclusive_scan(first, last, out, init, op);
    assert(std::equal(out, end, rFirst, rLast));

    // In place
    std::copy(first, last, out);
    end = std::exclusive_scan(out, end, out, init, op);
    assert(std::equal(out, end, rFirst, rLast));
}


template <class Iter>
TEST_CONSTEXPR_CXX20 void
test()
{
    int ia[]         = {1, 3, 5,  7,   9};
    const int pRes[] = {0, 1, 4,  9,  16};
    const int mRes[] = {1, 1, 3, 15, 105};
    const unsigned sa = sizeof(ia) / sizeof(ia[0]);
    static_assert(sa == sizeof(pRes) / sizeof(pRes[0]));       // just to be sure
    static_assert(sa == sizeof(mRes) / sizeof(mRes[0]));       // just to be sure

    for (unsigned int i = 0; i < sa; ++i ) {
        test(Iter(ia), Iter(ia + i), 0, std::plus<>(),       pRes, pRes + i);
        test(Iter(ia), Iter(ia + i), 1, std::multiplies<>(), mRes, mRes + i);
    }
}

TEST_CONSTEXPR_CXX20 bool
test()
{
//  All the iterator categories
    test<cpp17_input_iterator        <const int*> >();
    test<forward_iterator      <const int*> >();
    test<bidirectional_iterator<const int*> >();
    test<random_access_iterator<const int*> >();
    test<const int*>();
    test<      int*>();

//  Make sure that the calculations are done using the init typedef
    {
    std::array<unsigned char, 10> v;
    std::iota(v.begin(), v.end(), static_cast<unsigned char>(1));
    std::array<size_t, 10> res;
    std::exclusive_scan(v.begin(), v.end(), res.begin(), 1, std::multiplies<>());

    assert(res.size() == 10);
    size_t j = 1;
    assert(res[0] == 1);
    for (size_t i = 1; i < v.size(); ++i)
    {
        j *= i;
        assert(res[i] == j);
    }
    }

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
