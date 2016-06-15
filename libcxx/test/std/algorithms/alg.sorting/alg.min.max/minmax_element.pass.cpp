//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <algorithm>

// template<ForwardIterator Iter>
//   requires LessThanComparable<Iter::value_type>
//   pair<Iter, Iter>
//   minmax_element(Iter first, Iter last);

#include <algorithm>
#include <cassert>

#include "test_iterators.h"

template <class Iter>
void
test(Iter first, Iter last)
{
    std::pair<Iter, Iter> p = std::minmax_element(first, last);
    if (first != last)
    {
        for (Iter j = first; j != last; ++j)
        {
            assert(!(*j < *p.first));
            assert(!(*p.second < *j));
        }
    }
    else
    {
        assert(p.first == last);
        assert(p.second == last);
    }
}

template <class Iter>
void
test(unsigned N)
{
    int* a = new int[N];
    for (int i = 0; i < N; ++i)
        a[i] = i;
    std::random_shuffle(a, a+N);
    test(Iter(a), Iter(a+N));
    delete [] a;
}

template <class Iter>
void
test()
{
    test<Iter>(0);
    test<Iter>(1);
    test<Iter>(2);
    test<Iter>(3);
    test<Iter>(10);
    test<Iter>(1000);
    {
    const unsigned N = 100;
    int* a = new int[N];
    for (int i = 0; i < N; ++i)
        a[i] = 5;
    std::random_shuffle(a, a+N);
    std::pair<Iter, Iter> p = std::minmax_element(Iter(a), Iter(a+N));
    assert(base(p.first) == a);
    assert(base(p.second) == a+N-1);
    delete [] a;
    }
}

#if TEST_STD_VER >= 14
constexpr int il[] = { 2, 4, 6, 8, 7, 5, 3, 1 };
#endif

void constexpr_test()
{
#if TEST_STD_VER >= 14
    constexpr auto p = std::minmax_element(il, il+8);
    static_assert ( *(p.first)  == 1, "" );
    static_assert ( *(p.second) == 8, "" );
#endif
}

int main()
{
    test<forward_iterator<const int*> >();
    test<bidirectional_iterator<const int*> >();
    test<random_access_iterator<const int*> >();
    test<const int*>();

   constexpr_test();
}
