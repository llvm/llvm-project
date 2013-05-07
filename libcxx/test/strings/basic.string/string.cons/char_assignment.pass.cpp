//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <string>

// basic_string<charT,traits,Allocator>& operator=(charT c);

#include <string>
#include <cassert>

template <class S>
void
test(S s1, typename S::value_type s2)
{
    typedef typename S::traits_type T;
    s1 = s2;
    assert(s1.__invariants());
    assert(s1.size() == 1);
    assert(T::eq(s1[0], s2));
    assert(s1.capacity() >= s1.size());
}

int main()
{
    typedef std::string S;
    test(S(), 'a');
    test(S("1"), 'a');
    test(S("123456789"), 'a');
    test(S("1234567890123456789012345678901234567890123456789012345678901234567890"), 'a');
}
