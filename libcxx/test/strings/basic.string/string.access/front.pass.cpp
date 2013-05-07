//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <string>

// const charT& front() const;
//       charT& front();

#include <string>
#include <cassert>

template <class S>
void
test(S s)
{
    const S& cs = s;
    assert(&cs.front() == &cs[0]);
    assert(&s.front() == &s[0]);
    s.front() = typename S::value_type('z');
    assert(s.front() == typename S::value_type('z'));
}

int main()
{
    typedef std::string S;
    test(S("1"));
    test(S("1234567890123456789012345678901234567890"));
}
