//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <forward_list>

// forward_list();

#include <forward_list>
#include <cassert>

int main()
{
    {
        typedef int T;
        typedef std::forward_list<T> C;
        C c;
        assert(c.empty());
    }
}
