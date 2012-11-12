//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <string>

// basic_string(const basic_string<charT,traits,Allocator>& str,
//              size_type pos, size_type n = npos,
//              const Allocator& a = Allocator());

#include <string>
#include <stdexcept>
#include <algorithm>
#include <cassert>

#include "../test_allocator.h"

template <class S>
void
test(S str, unsigned pos)
{
    typedef typename S::traits_type T;
    typedef typename S::allocator_type A;
    try
    {
        S s2(str, pos);
        assert(s2.__invariants());
        assert(pos <= str.size());
        unsigned rlen = str.size() - pos;
        assert(s2.size() == rlen);
        assert(T::compare(s2.data(), str.data() + pos, rlen) == 0);
        assert(s2.get_allocator() == A());
        assert(s2.capacity() >= s2.size());
    }
    catch (std::out_of_range&)
    {
        assert(pos > str.size());
    }
}

template <class S>
void
test(S str, unsigned pos, unsigned n)
{
    typedef typename S::traits_type T;
    typedef typename S::allocator_type A;
    try
    {
        S s2(str, pos, n);
        assert(s2.__invariants());
        assert(pos <= str.size());
        unsigned rlen = std::min(str.size() - pos, n);
        assert(s2.size() == rlen);
        assert(T::compare(s2.data(), str.data() + pos, rlen) == 0);
        assert(s2.get_allocator() == A());
        assert(s2.capacity() >= s2.size());
    }
    catch (std::out_of_range&)
    {
        assert(pos > str.size());
    }
}

template <class S>
void
test(S str, unsigned pos, unsigned n, const typename S::allocator_type& a)
{
    typedef typename S::traits_type T;
    typedef typename S::allocator_type A;
    try
    {
        S s2(str, pos, n, a);
        assert(s2.__invariants());
        assert(pos <= str.size());
        unsigned rlen = std::min(str.size() - pos, n);
        assert(s2.size() == rlen);
        assert(T::compare(s2.data(), str.data() + pos, rlen) == 0);
        assert(s2.get_allocator() == a);
        assert(s2.capacity() >= s2.size());
    }
    catch (std::out_of_range&)
    {
        assert(pos > str.size());
    }
}

int main()
{
    typedef test_allocator<char> A;
    typedef std::basic_string<char, std::char_traits<char>, A> S;

    test(S(A(3)), 0);
    test(S(A(3)), 1);
    test(S("1", A(5)), 0);
    test(S("1", A(5)), 1);
    test(S("1", A(5)), 2);
    test(S("1234567890123456789012345678901234567890123456789012345678901234567890", A(7)), 0);
    test(S("1234567890123456789012345678901234567890123456789012345678901234567890", A(7)), 5);
    test(S("1234567890123456789012345678901234567890123456789012345678901234567890", A(7)), 50);
    test(S("1234567890123456789012345678901234567890123456789012345678901234567890", A(7)), 500);

    test(S(A(3)), 0, 0);
    test(S(A(3)), 0, 1);
    test(S(A(3)), 1, 0);
    test(S(A(3)), 1, 1);
    test(S(A(3)), 1, 2);
    test(S("1", A(5)), 0, 0);
    test(S("1", A(5)), 0, 1);
    test(S("1234567890123456789012345678901234567890123456789012345678901234567890", A(7)), 50, 0);
    test(S("1234567890123456789012345678901234567890123456789012345678901234567890", A(7)), 50, 1);
    test(S("1234567890123456789012345678901234567890123456789012345678901234567890", A(7)), 50, 10);
    test(S("1234567890123456789012345678901234567890123456789012345678901234567890", A(7)), 50, 100);

    test(S(A(3)), 0, 0, A(4));
    test(S(A(3)), 0, 1, A(4));
    test(S(A(3)), 1, 0, A(4));
    test(S(A(3)), 1, 1, A(4));
    test(S(A(3)), 1, 2, A(4));
    test(S("1", A(5)), 0, 0, A(6));
    test(S("1", A(5)), 0, 1, A(6));
    test(S("1234567890123456789012345678901234567890123456789012345678901234567890", A(7)), 50, 0, A(8));
    test(S("1234567890123456789012345678901234567890123456789012345678901234567890", A(7)), 50, 1, A(8));
    test(S("1234567890123456789012345678901234567890123456789012345678901234567890", A(7)), 50, 10, A(8));
    test(S("1234567890123456789012345678901234567890123456789012345678901234567890", A(7)), 50, 100, A(8));
}
