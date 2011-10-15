//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <string>

// void reserve(size_type res_arg=0);

#include <string>
#include <stdexcept>
#include <cassert>

template <class S>
void
test(S s)
{
    typename S::size_type old_cap = s.capacity();
    S s0 = s;
    s.reserve();
    assert(s.__invariants());
    assert(s == s0);
    assert(s.capacity() <= old_cap);
    assert(s.capacity() >= s.size());
}

template <class S>
void
test(S s, typename S::size_type res_arg)
{
    typename S::size_type old_cap = s.capacity();
    S s0 = s;
    try
    {
        s.reserve(res_arg);
        assert(res_arg <= s.max_size());
        assert(s == s0);
        assert(s.capacity() >= res_arg);
        assert(s.capacity() >= s.size());
    }
    catch (std::length_error&)
    {
        assert(res_arg > s.max_size());
    }
}

int main()
{
    typedef std::string S;
    {
    S s;
    test(s);

    s.assign(10, 'a');
    s.erase(5);
    test(s);

    s.assign(100, 'a');
    s.erase(50);
    test(s);
    }
    {
    S s;
    test(s, 5);
    test(s, 10);
    test(s, 50);
    }
    {
    S s(100, 'a');
    s.erase(50);
    test(s, 5);
    test(s, 10);
    test(s, 50);
    test(s, 100);
    test(s, S::npos);
    }
}
