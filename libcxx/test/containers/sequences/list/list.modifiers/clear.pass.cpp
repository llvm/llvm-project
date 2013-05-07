//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <list>

// void clear();

#include <list>
#include <cassert>

int main()
{
    int a[] = {1, 2, 3};
    std::list<int> c(a, a+3);
    c.clear();
    assert(c.empty());
}
