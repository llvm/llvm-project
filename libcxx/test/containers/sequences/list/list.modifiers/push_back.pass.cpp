//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <list>

// void push_back(const value_type& x);

#include <list>
#include <cassert>

int main()
{
    std::list<int> c;
    for (int i = 0; i < 5; ++i)
        c.push_back(i);
    int a[] = {0, 1, 2, 3, 4};
    assert(c == std::list<int>(a, a+5));
}
