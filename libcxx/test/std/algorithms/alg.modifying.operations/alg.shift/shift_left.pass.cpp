//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// <algorithm>

// template<class ForwardIterator>
// constexpr ForwardIterator
//   shift_left(ForwardIterator first, ForwardIterator last,
//              typename iterator_traits<ForwardIterator>::difference_type n);

#include <algorithm>
#include <cassert>

#include "test_macros.h"
#include "test_iterators.h"
#include "MoveOnly.h"

template<class T, class Iter>
constexpr bool test()
{
    int orig[] = {3,1,4,1,5, 9,2,6,5,3, 5,8,9,7,9};
    T work[] = {3,1,4,1,5, 9,2,6,5,3, 5,8,9,7,9};

    for (int n = 0; n <= 15; ++n) {
        for (int k = 0; k <= n+2; ++k) {
            std::copy(orig, orig+n, work);
            Iter it = std::shift_left(Iter(work), Iter(work+n), k);
            if (0 <= k && k < n) {
                assert(it == Iter(work+n-k));
                assert(std::equal(orig+k, orig+n, work, work+n-k));
            } else {
                assert(it == Iter(work));
                assert(std::equal(orig, orig+n, work, work+n));
            }
        }
    }

    // n == 0
    {
        T input[]          = { 0, 1, 2 };
        const T expected[] = { 0, 1, 2 };
        Iter b = Iter(std::begin(input));
        Iter e = Iter(std::end(input));
        Iter it = std::shift_left(b, e, 0);
        assert(std::equal(std::begin(expected), std::end(expected), b, e));
        assert(it == e);
    }

    // n > 0 && n < len
    {
        T input[]          = { 0, 1, 2 };
        const T expected[] = { 1, 2 };
        Iter b = Iter(std::begin(input));
        Iter e = Iter(std::end(input));
        Iter it = std::shift_left(b, e, 1);
        assert(std::equal(std::begin(expected), std::end(expected), b, it));
    }
    {
        T input[]          = { 1, 2, 3, 4, 5, 6, 7, 8 };
        const T expected[] = { 3, 4, 5, 6, 7, 8 };
        Iter b = Iter(std::begin(input));
        Iter e = Iter(std::end(input));
        Iter it = std::shift_left(b, e, 2);
        assert(std::equal(std::begin(expected), std::end(expected), b, it));
    }
    {
        T input[]          = { 1, 2, 3, 4, 5, 6, 7, 8 };
        const T expected[] = { 7, 8 };
        Iter b = Iter(std::begin(input));
        Iter e = Iter(std::end(input));
        Iter it = std::shift_left(b, e, 6);
        assert(std::equal(std::begin(expected), std::end(expected), b, it));
    }

    // n == len
    {
        T input[]          = { 0, 1, 2 };
        const T expected[] = { 0, 1, 2 };
        Iter b = Iter(std::begin(input));
        Iter e = Iter(std::end(input));
        Iter it = std::shift_left(b, e, std::size(input));
        assert(std::equal(std::begin(expected), std::end(expected), b, e));
        assert(it == b);
    }

    // n > len
    {
        T input[]          = { 0, 1, 2 };
        const T expected[] = { 0, 1, 2 };
        Iter b = Iter(std::begin(input));
        Iter e = Iter(std::end(input));
        Iter it = std::shift_left(b, e, std::size(input) + 1);
        assert(std::equal(std::begin(expected), std::end(expected), b, e));
        assert(it == b);
    }

    return true;
}

int main(int, char**)
{
    test<int, forward_iterator<int*>>();
    test<int, bidirectional_iterator<int*>>();
    test<int, random_access_iterator<int*>>();
    test<int, int*>();
    test<MoveOnly, forward_iterator<MoveOnly*>>();
    test<MoveOnly, bidirectional_iterator<MoveOnly*>>();
    test<MoveOnly, random_access_iterator<MoveOnly*>>();
    test<MoveOnly, MoveOnly*>();

    static_assert(test<int, forward_iterator<int*>>());
    static_assert(test<int, bidirectional_iterator<int*>>());
    static_assert(test<int, random_access_iterator<int*>>());
    static_assert(test<int, int*>());
    static_assert(test<MoveOnly, forward_iterator<MoveOnly*>>());
    static_assert(test<MoveOnly, bidirectional_iterator<MoveOnly*>>());
    static_assert(test<MoveOnly, random_access_iterator<MoveOnly*>>());
    static_assert(test<MoveOnly, MoveOnly*>());

    return 0;
}
