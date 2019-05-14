//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03

// <vector>

// template <class... Args> reference emplace_back(Args&&... args);
// return type is 'reference' in C++17; 'void' before

#include <vector>
#include <cassert>
#include "test_macros.h"
#include "test_allocator.h"
#include "min_allocator.h"
#include "test_allocator.h"
#include "asan_testing.h"

class A
{
    int i_;
    double d_;

    A(const A&);
    A& operator=(const A&);
public:
    A(int i, double d)
        : i_(i), d_(d) {}

    A(A&& a)
        : i_(a.i_),
          d_(a.d_)
    {
        a.i_ = 0;
        a.d_ = 0;
    }

    A& operator=(A&& a)
    {
        i_ = a.i_;
        d_ = a.d_;
        a.i_ = 0;
        a.d_ = 0;
        return *this;
    }

    int geti() const {return i_;}
    double getd() const {return d_;}
};

int main(int, char**)
{
    {
        std::vector<A> c;
#if TEST_STD_VER > 14
        A& r1 = c.emplace_back(2, 3.5);
        assert(c.size() == 1);
        assert(&r1 == &c.back());
        assert(c.front().geti() == 2);
        assert(c.front().getd() == 3.5);
        assert(is_contiguous_container_asan_correct(c));
        A& r2 = c.emplace_back(3, 4.5);
        assert(c.size() == 2);
        assert(&r2 == &c.back());
#else
        c.emplace_back(2, 3.5);
        assert(c.size() == 1);
        assert(c.front().geti() == 2);
        assert(c.front().getd() == 3.5);
        assert(is_contiguous_container_asan_correct(c));
        c.emplace_back(3, 4.5);
        assert(c.size() == 2);
#endif
        assert(c.front().geti() == 2);
        assert(c.front().getd() == 3.5);
        assert(c.back().geti() == 3);
        assert(c.back().getd() == 4.5);
        assert(is_contiguous_container_asan_correct(c));
    }
    {
        std::vector<A, limited_allocator<A, 4> > c;
#if TEST_STD_VER > 14
        A& r1 = c.emplace_back(2, 3.5);
        assert(c.size() == 1);
        assert(&r1 == &c.back());
        assert(c.front().geti() == 2);
        assert(c.front().getd() == 3.5);
        assert(is_contiguous_container_asan_correct(c));
        A& r2 = c.emplace_back(3, 4.5);
        assert(c.size() == 2);
        assert(&r2 == &c.back());
#else
        c.emplace_back(2, 3.5);
        assert(c.size() == 1);
        assert(c.front().geti() == 2);
        assert(c.front().getd() == 3.5);
        assert(is_contiguous_container_asan_correct(c));
        c.emplace_back(3, 4.5);
        assert(c.size() == 2);
#endif
        assert(c.front().geti() == 2);
        assert(c.front().getd() == 3.5);
        assert(c.back().geti() == 3);
        assert(c.back().getd() == 4.5);
        assert(is_contiguous_container_asan_correct(c));
    }
    {
        std::vector<A, min_allocator<A>> c;
#if TEST_STD_VER > 14
        A& r1 = c.emplace_back(2, 3.5);
        assert(c.size() == 1);
        assert(&r1 == &c.back());
        assert(c.front().geti() == 2);
        assert(c.front().getd() == 3.5);
        assert(is_contiguous_container_asan_correct(c));
        A& r2 = c.emplace_back(3, 4.5);
        assert(c.size() == 2);
        assert(&r2 == &c.back());
#else
        c.emplace_back(2, 3.5);
        assert(c.size() == 1);
        assert(c.front().geti() == 2);
        assert(c.front().getd() == 3.5);
        assert(is_contiguous_container_asan_correct(c));
        c.emplace_back(3, 4.5);
        assert(c.size() == 2);
#endif
        assert(c.front().geti() == 2);
        assert(c.front().getd() == 3.5);
        assert(c.back().geti() == 3);
        assert(c.back().getd() == 4.5);
        assert(is_contiguous_container_asan_correct(c));
    }
    {
        std::vector<Tag_X, TaggingAllocator<Tag_X>> c;
        c.emplace_back();
        assert(c.size() == 1);
        c.emplace_back(1, 2, 3);
        assert(c.size() == 2);
        assert(is_contiguous_container_asan_correct(c));
    }

    { // LWG 2164
        int arr[] = {0, 1, 2, 3, 4};
        int sz = 5;
        std::vector<int> c(arr, arr+sz);
        while (c.size() < c.capacity())
            c.push_back(sz++);
        c.emplace_back(c.front());
        assert(c.back() == 0);
        for (int i = 0; i < sz; ++i)
            assert(c[i] == i);
    }
  return 0;
}
