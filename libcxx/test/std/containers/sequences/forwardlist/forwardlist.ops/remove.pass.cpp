//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <forward_list>

// void remove(const value_type& v);      // C++17 and before
// size_type remove(const value_type& v); // C++20 and after

#include <forward_list>
#include <iterator>
#include <cassert>

#include "test_macros.h"
#include "min_allocator.h"

template <class L>
void do_remove(L &l, const typename L::value_type &value, typename L::size_type expected)
{
    typename L::size_type old_size = std::distance(l.begin(), l.end());
#if TEST_STD_VER > 17
    ASSERT_SAME_TYPE(decltype(l.remove(value)), typename L::size_type);
    assert(l.remove(value) == expected);
#else
    ASSERT_SAME_TYPE(decltype(l.remove(value)), void);
    l.remove(value);
#endif
    assert(old_size - std::distance(l.begin(), l.end()) == expected);
}


struct S {
    S(int i) : i_(new int(i)) {}
    S(const S &rhs) : i_(new int(*rhs.i_)) {}
    S& operator = (const S &rhs) { *i_ = *rhs.i_; return *this; }
    ~S () { delete i_; i_ = NULL; }
    bool operator == (const S &rhs) const { return *i_ == *rhs.i_; }
    int get () const { return *i_; }
    int *i_;
    };


int main(int, char**)
{
    {
        typedef int T;
        typedef std::forward_list<T> C;
        const T t1[] = {0, 5, 5, 0, 0, 0, 5};
        const T t2[] = {5, 5, 5};
        C c1(std::begin(t1), std::end(t1));
        C c2(std::begin(t2), std::end(t2));
        do_remove(c1, 0, 4);
        assert(c1 == c2);
    }
    {
        typedef int T;
        typedef std::forward_list<T> C;
        const T t1[] = {0, 0, 0, 0};
        C c1(std::begin(t1), std::end(t1));
        C c2;
        do_remove(c1, 0, 4);
        assert(c1 == c2);
    }
    {
        typedef int T;
        typedef std::forward_list<T> C;
        const T t1[] = {5, 5, 5};
        const T t2[] = {5, 5, 5};
        C c1(std::begin(t1), std::end(t1));
        C c2(std::begin(t2), std::end(t2));
        do_remove(c1, 0, 0);
        assert(c1 == c2);
    }
    {
        typedef int T;
        typedef std::forward_list<T> C;
        C c1;
        C c2;
        do_remove(c1, 0, 0);
        assert(c1 == c2);
    }
    {
        typedef int T;
        typedef std::forward_list<T> C;
        const T t1[] = {5, 5, 5, 0};
        const T t2[] = {5, 5, 5};
        C c1(std::begin(t1), std::end(t1));
        C c2(std::begin(t2), std::end(t2));
        do_remove(c1, 0, 1);
        assert(c1 == c2);
    }
    {  // LWG issue #526
    typedef int T;
    typedef std::forward_list<T> C;
    int t1[] = {1, 2, 1, 3, 5, 8, 11};
    int t2[] = {   2,    3, 5, 8, 11};
    C c1(std::begin(t1), std::end(t1));
    C c2(std::begin(t2), std::end(t2));
    do_remove(c1, c1.front(), 2);
    assert(c1 == c2);
    }
    {
    typedef S T;
    typedef std::forward_list<T> C;
    int t1[] = {1, 2, 1, 3, 5, 8, 11, 1};
    int t2[] = {   2,    3, 5, 8, 11   };
    C c;
    for(int *ip = std::end(t1); ip != std::begin(t1);)
        c.push_front(S(*--ip));
    do_remove(c, c.front(), 3);
    C::const_iterator it = c.begin();
    for(int *ip = std::begin(t2); ip != std::end(t2); ++ip, ++it) {
        assert ( it != c.end());
        assert ( *ip == it->get());
        }
    assert ( it == c.end ());
    }
#if TEST_STD_VER >= 11
    {
        typedef int T;
        typedef std::forward_list<T, min_allocator<T>> C;
        const T t1[] = {0, 5, 5, 0, 0, 0, 5};
        const T t2[] = {5, 5, 5};
        C c1(std::begin(t1), std::end(t1));
        C c2(std::begin(t2), std::end(t2));
        do_remove(c1, 0, 4);
        assert(c1 == c2);
    }
    {
        typedef int T;
        typedef std::forward_list<T, min_allocator<T>> C;
        const T t1[] = {0, 0, 0, 0};
        C c1(std::begin(t1), std::end(t1));
        C c2;
        do_remove(c1, 0, 4);
        assert(c1 == c2);
    }
    {
        typedef int T;
        typedef std::forward_list<T, min_allocator<T>> C;
        const T t1[] = {5, 5, 5};
        const T t2[] = {5, 5, 5};
        C c1(std::begin(t1), std::end(t1));
        C c2(std::begin(t2), std::end(t2));
        do_remove(c1, 0, 0);
        assert(c1 == c2);
    }
    {
        typedef int T;
        typedef std::forward_list<T, min_allocator<T>> C;
        C c1;
        C c2;
        do_remove(c1, 0, 0);
        assert(c1 == c2);
    }
    {
        typedef int T;
        typedef std::forward_list<T, min_allocator<T>> C;
        const T t1[] = {5, 5, 5, 0};
        const T t2[] = {5, 5, 5};
        C c1(std::begin(t1), std::end(t1));
        C c2(std::begin(t2), std::end(t2));
        do_remove(c1, 0, 1);
        assert(c1 == c2);
    }
#endif

  return 0;
}
