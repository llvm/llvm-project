//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <list>

// template <class Pred> void remove_if(Pred pred);

#include <list>
#include <cassert>
#include <functional>

bool g(int i)
{
    return i < 3;
}

int main()
{
    int a1[] = {1, 2, 3, 4};
    int a2[] = {3, 4};
    std::list<int> c(a1, a1+4);
    c.remove_if(g);
    assert(c == std::list<int>(a2, a2+2));
}
