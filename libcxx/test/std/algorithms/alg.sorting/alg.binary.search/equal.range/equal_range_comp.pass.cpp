//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <algorithm>

// template<ForwardIterator Iter, class T, CopyConstructible Compare>
//   requires Predicate<Compare, T, Iter::value_type>
//         && Predicate<Compare, Iter::value_type, T>
//   pair<Iter, Iter>
//   equal_range(Iter first, Iter last, const T& value, Compare comp);

#include <algorithm>
#include <functional>
#include <vector>
#include <cassert>
#include <cstddef>

#include "test_iterators.h"

template <class Iter, class T>
void
test(Iter first, Iter last, const T& value)
{
    std::pair<Iter, Iter> i = std::equal_range(first, last, value, std::greater<int>());
    for (Iter j = first; j != i.first; ++j)
        assert(std::greater<int>()(*j, value));
    for (Iter j = i.first; j != last; ++j)
        assert(!std::greater<int>()(*j, value));
    for (Iter j = first; j != i.second; ++j)
        assert(!std::greater<int>()(value, *j));
    for (Iter j = i.second; j != last; ++j)
        assert(std::greater<int>()(value, *j));
}

template <class Iter>
void
test()
{
    const unsigned N = 1000;
    const int M = 10;
    std::vector<int> v(N);
    int x = 0;
    for (std::size_t i = 0; i < v.size(); ++i)
    {
        v[i] = x;
        if (++x == M)
            x = 0;
    }
    std::sort(v.begin(), v.end(), std::greater<int>());
    for (x = 0; x <= M; ++x)
        test(Iter(v.data()), Iter(v.data()+v.size()), x);
}

int main()
{
    int d[] = {3, 2, 1, 0};
    for (int* e = d; e <= d+4; ++e)
        for (int x = -1; x <= 4; ++x)
            test(d, e, x);

    test<forward_iterator<const int*> >();
    test<bidirectional_iterator<const int*> >();
    test<random_access_iterator<const int*> >();
    test<const int*>();
}
