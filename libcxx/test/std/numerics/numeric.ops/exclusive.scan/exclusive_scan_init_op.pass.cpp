//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <numeric>
// UNSUPPORTED: c++03, c++11, c++14
// UNSUPPORTED: clang-8
// UNSUPPORTED: gcc-9

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
#include <vector>

#include "test_macros.h"
#include "test_iterators.h"
// FIXME Remove constexpr vector workaround introduced in D90569
#if TEST_STD_VER > 17
#include <span>
#endif

template <class Iter1, class T, class Op, class Iter2>
TEST_CONSTEXPR_CXX20 void
test(Iter1 first, Iter1 last, T init, Op op, Iter2 rFirst, Iter2 rLast)
{
    // C++17 doesn't test constexpr so can use a vector.
    // C++20 can use vector in constexpr evaluation, but both libc++ and MSVC
    // don't have the support yet. In these cases use a std::span for the test.
    // FIXME Remove constexpr vector workaround introduced in D90569
    size_t size = std::distance(first, last);
#if TEST_STD_VER < 20 || \
    (defined(__cpp_lib_constexpr_vector) && __cpp_lib_constexpr_vector >= 201907L)

    std::vector<typename std::iterator_traits<Iter1>::value_type> v(size);
#else
    assert((size <= 5) && "Increment the size of the array");
    typename std::iterator_traits<Iter1>::value_type b[5];
    std::span<typename std::iterator_traits<Iter1>::value_type> v{b, size};
#endif

//  Not in place
    std::exclusive_scan(first, last, v.begin(), init, op);
    assert(std::equal(v.begin(), v.end(), rFirst, rLast));

//  In place
    std::copy(first, last, v.begin());
    std::exclusive_scan(v.begin(), v.end(), v.begin(), init, op);
    assert(std::equal(v.begin(), v.end(), rFirst, rLast));
}


template <class Iter>
TEST_CONSTEXPR_CXX20 void
test()
{
          int ia[]   = {1, 3, 5,  7,   9};
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
    test<input_iterator        <const int*> >();
    test<forward_iterator      <const int*> >();
    test<bidirectional_iterator<const int*> >();
    test<random_access_iterator<const int*> >();
    test<const int*>();
    test<      int*>();

//  Make sure that the calculations are done using the init typedef
    {
    std::array<unsigned char, 10> v;
    std::iota(v.begin(), v.end(), static_cast<unsigned char>(1));
    // C++17 doesn't test constexpr so can use a vector.
    // C++20 can use vector in constexpr evaluation, but both libc++ and MSVC
    // don't have the support yet. In these cases use a std::span for the test.
    // FIXME Remove constexpr vector workaround introduced in D90569
#if TEST_STD_VER < 20 || \
    (defined(__cpp_lib_constexpr_vector) && __cpp_lib_constexpr_vector >= 201907L)
    std::vector<size_t> res;
    std::exclusive_scan(v.begin(), v.end(), std::back_inserter(res), 1, std::multiplies<>());
#else
    std::array<size_t, 10> res;
    std::exclusive_scan(v.begin(), v.end(), res.begin(), 1, std::multiplies<>());
#endif

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
