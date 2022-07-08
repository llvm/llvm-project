//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <iterator>

//   All of these became constexpr in C++17
//
// template <InputIterator Iter, class Distance>
//   constexpr void advance(Iter& i, Distance n);
//
// template <BidirectionalIterator Iter, class Distance>
//   constexpr void advance(Iter& i, Distance n);
//
// template <RandomAccessIterator Iter, class Distance>
//   constexpr void advance(Iter& i, Distance n);


// TODO: test_iterators.h includes <ranges>, and <ranges> includes <chrono> and <atomic>.
// Lots of implementation headers under <__chrono/> and <atomic> has signed to unsigned conversion,
// which will trigger the -Wsign-conversion warning.
// Once those headers are fixed, enable the -Wsign-conversion for this test by removing
// <TODO:Remove brackets> below

// Make sure we catch forced conversions to the difference_type if they happen.
// ADDITIONAL_COMPILE_FLAGS<TODO:Remove brackets>: -Wsign-conversion

#include <iterator>
#include <cassert>
#include <cstddef>
#include <type_traits>

#include "test_macros.h"
#include "test_iterators.h"

template <class Distance, class It>
TEST_CONSTEXPR_CXX17
void check_advance(It it, Distance n, It result)
{
    static_assert(std::is_same<decltype(std::advance(it, n)), void>::value, "");
    std::advance(it, n);
    assert(it == result);
}

TEST_CONSTEXPR_CXX17 bool tests()
{
    const char* s = "1234567890";

    // Check with iterator_traits::difference_type
    {
        typedef std::iterator_traits<const char*>::difference_type Distance;
        check_advance<Distance>(cpp17_input_iterator<const char*>(s), 10, cpp17_input_iterator<const char*>(s+10));
        check_advance<Distance>(forward_iterator<const char*>(s), 10, forward_iterator<const char*>(s+10));
        check_advance<Distance>(bidirectional_iterator<const char*>(s+5), 5, bidirectional_iterator<const char*>(s+10));
        check_advance<Distance>(bidirectional_iterator<const char*>(s+5), -5, bidirectional_iterator<const char*>(s));
        check_advance<Distance>(random_access_iterator<const char*>(s+5), 5, random_access_iterator<const char*>(s+10));
        check_advance<Distance>(random_access_iterator<const char*>(s+5), -5, random_access_iterator<const char*>(s));
        check_advance<Distance>(s+5, 5, s+10);
        check_advance<Distance>(s+5, -5, s);
    }

    // Also check with other distance types
    {
        typedef int Distance;
        check_advance<Distance>(cpp17_input_iterator<const char*>(s), 10, cpp17_input_iterator<const char*>(s+10));
        check_advance<Distance>(forward_iterator<const char*>(s), 10, forward_iterator<const char*>(s+10));
        check_advance<Distance>(bidirectional_iterator<const char*>(s), 10, bidirectional_iterator<const char*>(s+10));
        check_advance<Distance>(random_access_iterator<const char*>(s), 10, random_access_iterator<const char*>(s+10));
    }

    // Check with unsigned distance types to catch signedness-change issues
    {
        typedef std::size_t Distance;
        check_advance<Distance>(cpp17_input_iterator<const char*>(s), 10u, cpp17_input_iterator<const char*>(s+10));
        check_advance<Distance>(forward_iterator<const char*>(s), 10u, forward_iterator<const char*>(s+10));
        check_advance<Distance>(bidirectional_iterator<const char*>(s), 10u, bidirectional_iterator<const char*>(s+10));
        check_advance<Distance>(random_access_iterator<const char*>(s), 10u, random_access_iterator<const char*>(s+10));
    }

    return true;
}

int main(int, char**)
{
    tests();
#if TEST_STD_VER >= 17
    static_assert(tests(), "");
#endif
    return 0;
}
