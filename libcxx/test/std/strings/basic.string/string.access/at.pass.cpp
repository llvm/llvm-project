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

// const_reference at(size_type pos) const;
//       reference at(size_type pos);

#include <string>
#include <stdexcept>
#include <cassert>

#include "min_allocator.h"

template <class S>
void
test(S s, typename S::size_type pos)
{
    try
    {
        const S& cs = s;
        assert(s.at(pos) == s[pos]);
        assert(cs.at(pos) == cs[pos]);
        assert(pos < cs.size());
    }
    catch (std::out_of_range&)
    {
        assert(pos >= s.size());
    }
}

int main()
{
    {
    typedef std::string S;
    test(S(), 0);
    test(S("123"), 0);
    test(S("123"), 1);
    test(S("123"), 2);
    test(S("123"), 3);
    }
#if __cplusplus >= 201103L
    {
    typedef std::basic_string<char, std::char_traits<char>, min_allocator<char>> S;
    test(S(), 0);
    test(S("123"), 0);
    test(S("123"), 1);
    test(S("123"), 2);
    test(S("123"), 3);
    }
#endif
}
