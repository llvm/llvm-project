//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <string>

//       reverse_iterator rbegin();
// const_reverse_iterator rbegin() const;

#include <string>
#include <cassert>

template <class S>
void
test(S s)
{
    const S& cs = s;
    typename S::reverse_iterator b = s.rbegin();
    typename S::const_reverse_iterator cb = cs.rbegin();
    if (!s.empty())
    {
        assert(*b == s.back());
    }
    assert(b == cb);
}

int main()
{
    typedef std::string S;
    test(S());
    test(S("123"));
}
