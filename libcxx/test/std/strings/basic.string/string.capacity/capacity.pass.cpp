//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// XFAIL: libcpp-no-exceptions
// <string>

// size_type capacity() const;

#include <string>
#include <cassert>

#include "test_allocator.h"
#include "min_allocator.h"

template <class S>
void
test(S s)
{
    S::allocator_type::throw_after = 0;
    try
    {
        while (s.size() < s.capacity())
            s.push_back(typename S::value_type());
        assert(s.size() == s.capacity());
    }
    catch (...)
    {
        assert(false);
    }
    S::allocator_type::throw_after = INT_MAX;
}

int main()
{
    {
    typedef std::basic_string<char, std::char_traits<char>, test_allocator<char> > S;
    S s;
    test(s);
    s.assign(10, 'a');
    s.erase(5);
    test(s);
    s.assign(100, 'a');
    s.erase(50);
    test(s);
    }
#if __cplusplus >= 201103L
    {
    typedef std::basic_string<char, std::char_traits<char>, min_allocator<char>> S;
    S s;
    assert(s.capacity() > 0);
    }
#endif
}
