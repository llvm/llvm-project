//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <algorithm>

// template<RandomAccessIterator Iter, StrictWeakOrder<auto, Iter::value_type> Compare>
//   requires ShuffleIterator<Iter> && CopyConstructible<Compare>
//   void
//   sort_heap(Iter first, Iter last, Compare comp);

#include <algorithm>
#include <functional>
#include <random>
#include <cassert>
#ifndef _LIBCPP_HAS_NO_RVALUE_REFERENCES
#include <memory>

struct indirect_less
{
    template <class P>
    bool operator()(const P& x, const P& y)
        {return *x < *y;}
};

#endif  // _LIBCPP_HAS_NO_RVALUE_REFERENCES

std::mt19937 randomness;

void test(int N)
{
    int* ia = new int [N];
    for (int i = 0; i < N; ++i)
        ia[i] = i;
    std::shuffle(ia, ia+N, randomness);
    std::make_heap(ia, ia+N, std::greater<int>());
    std::sort_heap(ia, ia+N, std::greater<int>());
    assert(std::is_sorted(ia, ia+N, std::greater<int>()));
    delete [] ia;
}

int main()
{
    test(0);
    test(1);
    test(2);
    test(3);
    test(10);
    test(1000);

#ifndef _LIBCPP_HAS_NO_RVALUE_REFERENCES
    {
    const int N = 1000;
    std::unique_ptr<int>* ia = new std::unique_ptr<int> [N];
    for (int i = 0; i < N; ++i)
        ia[i].reset(new int(i));
    std::shuffle(ia, ia+N, randomness);
    std::make_heap(ia, ia+N, indirect_less());
    std::sort_heap(ia, ia+N, indirect_less());
    assert(std::is_sorted(ia, ia+N, indirect_less()));
    delete [] ia;
    }
#endif  // _LIBCPP_HAS_NO_RVALUE_REFERENCES
}
