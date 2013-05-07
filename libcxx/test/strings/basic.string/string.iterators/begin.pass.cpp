//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <string>

//       iterator begin();
// const_iterator begin() const;

#include <string>
#include <cassert>

template <class S>
void
test(S s)
{
    const S& cs = s;
    typename S::iterator b = s.begin();
    typename S::const_iterator cb = cs.begin();
    if (!s.empty())
    {
        assert(*b == s[0]);
    }
    assert(b == cb);
}

int main()
{
    typedef std::string S;
    test(S());
    test(S("123"));
}
