//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <list>

// template <class... Args> void emplace(const_iterator p, Args&&... args);

#if _LIBCPP_DEBUG >= 1
#define _LIBCPP_ASSERT(x, m) ((x) ? (void)0 : std::exit(0))
#endif

#include <list>
#include <cassert>

#include "min_allocator.h"

class A
{
    int i_;
    double d_;

    A(const A&);
    A& operator=(const A&);
public:
    A(int i, double d)
        : i_(i), d_(d) {}

    int geti() const {return i_;}
    double getd() const {return d_;}
};

int main()
{
#ifndef _LIBCPP_HAS_NO_RVALUE_REFERENCES
    {
    std::list<A> c;
    c.emplace(c.cbegin(), 2, 3.5);
    assert(c.size() == 1);
    assert(c.front().geti() == 2);
    assert(c.front().getd() == 3.5);
    c.emplace(c.cend(), 3, 4.5);
    assert(c.size() == 2);
    assert(c.front().geti() == 2);
    assert(c.front().getd() == 3.5);
    assert(c.back().geti() == 3);
    assert(c.back().getd() == 4.5);
    }
#if _LIBCPP_DEBUG >= 1
    {
        std::list<A> c1;
        std::list<A> c2;
        std::list<A>::iterator i = c1.emplace(c2.cbegin(), 2, 3.5);
        assert(false);
    }
#endif
#endif  // _LIBCPP_HAS_NO_RVALUE_REFERENCES
#if TEST_STD_VER >= 11
#ifndef _LIBCPP_HAS_NO_RVALUE_REFERENCES
    {
    std::list<A, min_allocator<A>> c;
    c.emplace(c.cbegin(), 2, 3.5);
    assert(c.size() == 1);
    assert(c.front().geti() == 2);
    assert(c.front().getd() == 3.5);
    c.emplace(c.cend(), 3, 4.5);
    assert(c.size() == 2);
    assert(c.front().geti() == 2);
    assert(c.front().getd() == 3.5);
    assert(c.back().geti() == 3);
    assert(c.back().getd() == 4.5);
    }
#if _LIBCPP_DEBUG >= 1
    {
        std::list<A, min_allocator<A>> c1;
        std::list<A, min_allocator<A>> c2;
        std::list<A, min_allocator<A>>::iterator i = c1.emplace(c2.cbegin(), 2, 3.5);
        assert(false);
    }
#endif
#endif  // _LIBCPP_HAS_NO_RVALUE_REFERENCES
#endif
}
