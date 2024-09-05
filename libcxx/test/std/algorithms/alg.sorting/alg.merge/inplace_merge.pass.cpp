//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <algorithm>

// template<BidirectionalIterator Iter>
//   requires ShuffleIterator<Iter>
//         && LessThanComparable<Iter::value_type>
//   void
//   inplace_merge(Iter first, Iter middle, Iter last);

#include <algorithm>
#include <cassert>
#include <random>
#include <vector>

#include "count_new.h"
#include "test_iterators.h"
#include "test_macros.h"

#if TEST_STD_VER >= 11
struct S {
    S() : i_(0) {}
    S(int i) : i_(i) {}

    S(const S&  rhs) : i_(rhs.i_) {}
    S(      S&& rhs) : i_(rhs.i_) { rhs.i_ = -1; }

    S& operator =(const S&  rhs) { i_ = rhs.i_;              return *this; }
    S& operator =(      S&& rhs) { i_ = rhs.i_; rhs.i_ = -2; assert(this != &rhs); return *this; }
    S& operator =(int i)         { i_ = i;                   return *this; }

    bool operator  <(const S&  rhs) const { return i_ < rhs.i_; }
    bool operator ==(const S&  rhs) const { return i_ == rhs.i_; }
    bool operator ==(int i)         const { return i_ == i; }

    void set(int i) { i_ = i; }

    int i_;
    };
#endif

std::mt19937 randomness;

template <class Iter>
void
test_one(unsigned N, unsigned M)
{
    typedef typename std::iterator_traits<Iter>::value_type value_type;
    assert(M <= N);
    value_type* ia = new value_type[N];
    for (unsigned i = 0; i < N; ++i)
        ia[i] = i;
    std::shuffle(ia, ia+N, randomness);
    std::sort(ia, ia+M);
    std::sort(ia+M, ia+N);
    std::inplace_merge(Iter(ia), Iter(ia+M), Iter(ia+N));
    if(N > 0)
    {
        assert(ia[0] == 0);
        assert(ia[N-1] == static_cast<value_type>(N-1));
        assert(std::is_sorted(ia, ia+N));
    }
    delete [] ia;
}

template <class Iter>
void
test(unsigned N)
{
    test_one<Iter>(N, 0);
    test_one<Iter>(N, N/4);
    test_one<Iter>(N, N/2);
    test_one<Iter>(N, 3*N/4);
    test_one<Iter>(N, N);
}

template <class Iter>
void
test()
{
    test_one<Iter>(0, 0);
    test_one<Iter>(1, 0);
    test_one<Iter>(1, 1);
    test_one<Iter>(2, 0);
    test_one<Iter>(2, 1);
    test_one<Iter>(2, 2);
    test_one<Iter>(3, 0);
    test_one<Iter>(3, 1);
    test_one<Iter>(3, 2);
    test_one<Iter>(3, 3);
    test<Iter>(4);
    test<Iter>(100);
    test<Iter>(1000);
}

int main(int, char**)
{
    test<bidirectional_iterator<int*> >();
    test<random_access_iterator<int*> >();
    test<int*>();

#if TEST_STD_VER >= 11
    test<bidirectional_iterator<S*> >();
    test<random_access_iterator<S*> >();
    test<S*>();
#endif

#if TEST_STD_VER >= 11 && !defined(TEST_HAS_NO_EXCEPTIONS)
    {
        std::vector<int> vec(150, 3);
        getGlobalMemCounter()->throw_after = 0;
        std::inplace_merge(vec.begin(), vec.begin() + 100, vec.end());
        assert(std::all_of(vec.begin(), vec.end(), [](int i) { return i == 3; }));
    }
#endif // TEST_STD_VER >= 11 && !defined(TEST_HAS_NO_EXCEPTIONS)

  return 0;
}
