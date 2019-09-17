//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <list>

// template <class Pred> void      remove_if(Pred pred); // before C++20
// template <class Pred> size_type remove_if(Pred pred); // c++20 and later

#include <list>
#include <cassert>
#include <functional>

#include "test_macros.h"
#include "min_allocator.h"
#include "counting_predicates.h"

bool even(int i)
{
    return i % 2 == 0;
}

bool g(int i)
{
    return i < 3;
}

struct PredLWG526 {
    PredLWG526 (int i) : i_(i) {};
    ~PredLWG526() { i_ = -32767; }
    bool operator() (const PredLWG526 &p) const { return p.i_ == i_; }

    bool operator==(int i) const { return i == i_;}
    int i_;
};

typedef unary_counting_predicate<bool(*)(int), int> Predicate;

int main(int, char**)
{
    {
    int a1[] = {1, 2, 3, 4};
    int a2[] = {3, 4};
    typedef std::list<int> L;
    L c(a1, a1+4);
    Predicate cp(g);
#if TEST_STD_VER > 17
	ASSERT_SAME_TYPE(L::size_type, decltype(c.remove_if(std::ref(cp))));
    assert(c.remove_if(std::ref(cp)) == 2);
#else
	ASSERT_SAME_TYPE(void, decltype(c.remove_if(std::ref(cp))));
    c.remove_if(std::ref(cp));
#endif
    assert(c == std::list<int>(a2, a2+2));
    assert(cp.count() == 4);
    }
    {
    int a1[] = {1, 2, 3, 4};
    int a2[] = {1, 3};
    std::list<int> c(a1, a1+4);
    Predicate cp(even);
#if TEST_STD_VER > 17
    assert(c.remove_if(std::ref(cp)) == 2);
#else
    c.remove_if(std::ref(cp));
#endif
    assert(c == std::list<int>(a2, a2+2));
    assert(cp.count() == 4);
    }
    { // LWG issue #526
    int a1[] = {1, 2, 1, 3, 5, 8, 11};
    int a2[] = {2, 3, 5, 8, 11};
    std::list<PredLWG526> c(a1, a1 + 7);
    c.remove_if(std::ref(c.front()));
    assert(c.size() == 5);
    for (size_t i = 0; i < c.size(); ++i)
    {
        assert(c.front() == a2[i]);
        c.pop_front();
    }
    }

#if TEST_STD_VER >= 11
    {
    int a1[] = {1, 2, 3, 4};
    int a2[] = {3, 4};
    std::list<int, min_allocator<int>> c(a1, a1+4);
    Predicate cp(g);
#if TEST_STD_VER > 17
    assert(c.remove_if(std::ref(cp)) == 2);
#else
    c.remove_if(std::ref(cp));
#endif
    assert((c == std::list<int, min_allocator<int>>(a2, a2+2)));
    assert(cp.count() == 4);
    }
#endif

  return 0;
}
