//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <string>

// const_reverse_iterator crend() const;

#include <string>
#include <cassert>

template <class S>
void
test(const S& s)
{
    typename S::const_reverse_iterator ce = s.crend();
    assert(ce == s.rend());
}

int main()
{
    typedef std::string S;
    test(S());
    test(S("123"));
}
