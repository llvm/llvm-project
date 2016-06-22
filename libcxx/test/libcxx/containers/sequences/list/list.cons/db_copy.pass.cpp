//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <list>

// list(list&& c);

#define _LIBCPP_DEBUG 1
#define _LIBCPP_ASSERT(x, m) ((x) ? (void)0 : std::exit(0))

#include <list>
#include <cstdlib>
#include <cassert>

int main()
{
    std::list<int> l1;
    l1.push_back(1); l1.push_back(2); l1.push_back(3);
    std::list<int>::iterator i = l1.begin();
    std::list<int> l2 = l1;
    l2.erase(i);
    assert(false);
}
