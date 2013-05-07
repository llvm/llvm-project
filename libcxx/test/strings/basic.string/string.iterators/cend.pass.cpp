//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <string>

// const_iterator cend() const;

#include <string>
#include <cassert>

template <class S>
void
test(const S& s)
{
    typename S::const_iterator ce = s.cend();
    assert(ce == s.end());
}

int main()
{
    typedef std::string S;
    test(S());
    test(S("123"));
}
