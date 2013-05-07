//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <string>

// void shrink_to_fit();

#include <string>
#include <cassert>

template <class S>
void
test(S s)
{
    typename S::size_type old_cap = s.capacity();
    S s0 = s;
    s.shrink_to_fit();
    assert(s.__invariants());
    assert(s == s0);
    assert(s.capacity() <= old_cap);
    assert(s.capacity() >= s.size());
}

int main()
{
    typedef std::string S;
    S s;
    test(s);

    s.assign(10, 'a');
    s.erase(5);
    test(s);

    s.assign(100, 'a');
    s.erase(50);
    test(s);
}
