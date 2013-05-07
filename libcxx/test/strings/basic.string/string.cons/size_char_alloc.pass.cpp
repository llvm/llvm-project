//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <string>

// basic_string(size_type n, charT c, const Allocator& a = Allocator());

#include <string>
#include <stdexcept>
#include <algorithm>
#include <cassert>

#include "../test_allocator.h"

template <class charT>
void
test(unsigned n, charT c)
{
    typedef std::basic_string<charT, std::char_traits<charT>, test_allocator<charT> > S;
    typedef typename S::traits_type T;
    typedef typename S::allocator_type A;
    S s2(n, c);
    assert(s2.__invariants());
    assert(s2.size() == n);
    for (unsigned i = 0; i < n; ++i)
        assert(s2[i] == c);
    assert(s2.get_allocator() == A());
    assert(s2.capacity() >= s2.size());
}

template <class charT>
void
test(unsigned n, charT c, const test_allocator<charT>& a)
{
    typedef std::basic_string<charT, std::char_traits<charT>, test_allocator<charT> > S;
    typedef typename S::traits_type T;
    typedef typename S::allocator_type A;
    S s2(n, c, a);
    assert(s2.__invariants());
    assert(s2.size() == n);
    for (unsigned i = 0; i < n; ++i)
        assert(s2[i] == c);
    assert(s2.get_allocator() == a);
    assert(s2.capacity() >= s2.size());
}

template <class Tp>
void
test(Tp n, Tp c)
{
    typedef char charT;
    typedef std::basic_string<charT, std::char_traits<charT>, test_allocator<charT> > S;
    typedef typename S::traits_type T;
    typedef typename S::allocator_type A;
    S s2(n, c);
    assert(s2.__invariants());
    assert(s2.size() == n);
    for (unsigned i = 0; i < n; ++i)
        assert(s2[i] == c);
    assert(s2.get_allocator() == A());
    assert(s2.capacity() >= s2.size());
}

template <class Tp>
void
test(Tp n, Tp c, const test_allocator<char>& a)
{
    typedef char charT;
    typedef std::basic_string<charT, std::char_traits<charT>, test_allocator<charT> > S;
    typedef typename S::traits_type T;
    typedef typename S::allocator_type A;
    S s2(n, c, a);
    assert(s2.__invariants());
    assert(s2.size() == n);
    for (unsigned i = 0; i < n; ++i)
        assert(s2[i] == c);
    assert(s2.get_allocator() == a);
    assert(s2.capacity() >= s2.size());
}

int main()
{
    typedef test_allocator<char> A;
    typedef std::basic_string<char, std::char_traits<char>, A> S;

    test(0, 'a');
    test(0, 'a', A(2));

    test(1, 'a');
    test(1, 'a', A(2));

    test(10, 'a');
    test(10, 'a', A(2));

    test(100, 'a');
    test(100, 'a', A(2));

    test(100, 65);
    test(100, 65, A(3));
}
