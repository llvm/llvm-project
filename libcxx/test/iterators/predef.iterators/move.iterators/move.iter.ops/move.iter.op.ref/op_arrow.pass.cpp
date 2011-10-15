//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <iterator>

// move_iterator

// pointer operator->() const;

#include <iterator>
#include <cassert>

template <class It>
void
test(It i)
{
    std::move_iterator<It> r(i);
    assert(r.operator->() == i);
}

int main()
{
    char s[] = "123";
    test(s);
}
