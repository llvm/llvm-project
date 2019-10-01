//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <list>

// template <class BinaryPred> void unique(BinaryPred pred);

#include <list>
#include <cassert>

#include "test_macros.h"
#include "min_allocator.h"

bool g(int x, int y)
{
    return x == y;
}

struct PredLWG529 {
    PredLWG529 (int i) : i_(i) {};
    ~PredLWG529() { i_ = -32767; }
    bool operator() (const PredLWG529 &lhs, const PredLWG529 &rhs) const { return lhs.i_ == rhs.i_; }

    bool operator==(int i) const { return i == i_;}
    int i_;
};

int main(int, char**)
{
    {
    int a1[] = {2, 1, 1, 4, 4, 4, 4, 3, 3};
    int a2[] = {2, 1, 4, 3};
    std::list<int> c(a1, a1+sizeof(a1)/sizeof(a1[0]));
    c.unique(g);
    assert(c == std::list<int>(a2, a2+4));
    }

    { // LWG issue #526
    int a1[] = {1, 1, 1, 2, 3, 5, 5, 2, 11};
    int a2[] = {1,       2, 3, 5,    2, 11};
    std::list<PredLWG529> c(a1, a1 + 9);
    c.unique(std::ref(c.front()));
    assert(c.size() == 6);
    for (size_t i = 0; i < c.size(); ++i)
    {
        assert(c.front() == a2[i]);
        c.pop_front();
    }
    }

#if TEST_STD_VER >= 11
    {
    int a1[] = {2, 1, 1, 4, 4, 4, 4, 3, 3};
    int a2[] = {2, 1, 4, 3};
    std::list<int, min_allocator<int>> c(a1, a1+sizeof(a1)/sizeof(a1[0]));
    c.unique(g);
    assert((c == std::list<int, min_allocator<int>>(a2, a2+4)));
    }
#endif

  return 0;
}
