//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <string>

// const_reference operator[](size_type pos) const;
//       reference operator[](size_type pos);

#include <string>
#include <cassert>

int main()
{
    typedef std::string S;
    S s("0123456789");
    const S& cs = s;
    for (S::size_type i = 0; i < cs.size(); ++i)
    {
        assert(s[i] == '0' + i);
        assert(cs[i] == s[i]);
    }
    assert(cs[cs.size()] == '\0');
    const S s2 = S();
    assert(s2[0] == '\0');
}
