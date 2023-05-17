//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <forward_list>

// template <class BinaryPredicate> void unique(BinaryPredicate binary_pred);      // C++17 and before
// template <class BinaryPredicate> size_type unique(BinaryPredicate binary_pred); // C++20 and after

#include <cassert>
#include <forward_list>
#include <functional>
#include <iterator>

#include "test_macros.h"
#include "min_allocator.h"

template <class L, class Predicate>
void do_unique(L &l, Predicate pred, typename L::size_type expected)
{
    typename L::size_type old_size = std::distance(l.begin(), l.end());
#if TEST_STD_VER > 17
    ASSERT_SAME_TYPE(decltype(l.unique(pred)), typename L::size_type);
    assert(l.unique(pred) == expected);
#else
    ASSERT_SAME_TYPE(decltype(l.unique(pred)), void);
    l.unique(pred);
#endif
    assert(old_size - std::distance(l.begin(), l.end()) == expected);
}


struct PredLWG526 {
    PredLWG526 (int i) : i_(i) {};
    ~PredLWG526() { i_ = -32767; }
    bool operator() (const PredLWG526 &lhs, const PredLWG526 &rhs) const { return lhs.i_ == rhs.i_; }

    bool operator==(int i) const { return i == i_;}
    int i_;
};


bool g(int x, int y)
{
    return x == y;
}

int main(int, char**)
{
    {
        typedef int T;
        typedef std::forward_list<T> C;
        const T t1[] = {0, 5, 5, 0, 0, 0, 5};
        const T t2[] = {0, 5, 0, 5};
        C c1(std::begin(t1), std::end(t1));
        C c2(std::begin(t2), std::end(t2));
        do_unique(c1, g, 3);
        assert(c1 == c2);
    }
    {
        typedef int T;
        typedef std::forward_list<T> C;
        const T t1[] = {0, 0, 0, 0};
        const T t2[] = {0};
        C c1(std::begin(t1), std::end(t1));
        C c2(std::begin(t2), std::end(t2));
        do_unique(c1, g, 3);
        assert(c1 == c2);
    }
    {
        typedef int T;
        typedef std::forward_list<T> C;
        const T t1[] = {5, 5, 5};
        const T t2[] = {5};
        C c1(std::begin(t1), std::end(t1));
        C c2(std::begin(t2), std::end(t2));
        do_unique(c1, g, 2);
        assert(c1 == c2);
    }
    {
        typedef int T;
        typedef std::forward_list<T> C;
        C c1;
        C c2;
        do_unique(c1, g, 0);
        assert(c1 == c2);
    }
    {
        typedef int T;
        typedef std::forward_list<T> C;
        const T t1[] = {5, 5, 5, 0};
        const T t2[] = {5, 0};
        C c1(std::begin(t1), std::end(t1));
        C c2(std::begin(t2), std::end(t2));
        do_unique(c1, g, 2);
        assert(c1 == c2);
    }

    { // LWG issue #526
    int a1[] = {1, 1, 1, 2, 3, 5, 2, 11};
    int a2[] = {1,       2, 3, 5, 2, 11};
    std::forward_list<PredLWG526> c1(a1, a1 + 8);
    do_unique(c1, std::ref(c1.front()), 2);
    for (std::size_t i = 0; i < 6; ++i)
    {
        assert(!c1.empty());
        assert(c1.front() == a2[i]);
        c1.pop_front();
    }
    assert(c1.empty());
    }

#if TEST_STD_VER >= 11
    {
        typedef int T;
        typedef std::forward_list<T, min_allocator<T>> C;
        const T t1[] = {0, 5, 5, 0, 0, 0, 5};
        const T t2[] = {0, 5, 0, 5};
        C c1(std::begin(t1), std::end(t1));
        C c2(std::begin(t2), std::end(t2));
        do_unique(c1, g, 3);
        assert(c1 == c2);
    }
    {
        typedef int T;
        typedef std::forward_list<T, min_allocator<T>> C;
        const T t1[] = {0, 0, 0, 0};
        const T t2[] = {0};
        C c1(std::begin(t1), std::end(t1));
        C c2(std::begin(t2), std::end(t2));
        do_unique(c1, g, 3);
        assert(c1 == c2);
    }
    {
        typedef int T;
        typedef std::forward_list<T, min_allocator<T>> C;
        const T t1[] = {5, 5, 5};
        const T t2[] = {5};
        C c1(std::begin(t1), std::end(t1));
        C c2(std::begin(t2), std::end(t2));
        do_unique(c1, g, 2);
        assert(c1 == c2);
    }
    {
        typedef int T;
        typedef std::forward_list<T, min_allocator<T>> C;
        C c1;
        C c2;
        do_unique(c1, g, 0);
        assert(c1 == c2);
    }
    {
        typedef int T;
        typedef std::forward_list<T, min_allocator<T>> C;
        const T t1[] = {5, 5, 5, 0};
        const T t2[] = {5, 0};
        C c1(std::begin(t1), std::end(t1));
        C c2(std::begin(t2), std::end(t2));
        do_unique(c1, g, 2);
        assert(c1 == c2);
    }
#endif

  return 0;
}
