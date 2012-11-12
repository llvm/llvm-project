//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <string>

// bool empty() const;

#include <string>
#include <cassert>

template <class S>
void
test(const S& s)
{
    assert(s.empty() == (s.size() == 0));
}

int main()
{
    typedef std::string S;
    test(S());
    test(S("123"));
    test(S("12345678901234567890123456789012345678901234567890"));
}
