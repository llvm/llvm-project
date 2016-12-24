//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <list>

// Dereference non-dereferenceable iterator.

#define _LIBCPP_DEBUG 1
#define _LIBCPP_ASSERT(x, m) ((x) ? (void)0 : std::exit(0))

#include <list>
#include <cassert>
#include <iterator>
#include <exception>
#include <cstdlib>

int main()
{
    typedef int T;
    typedef std::list<T> C;
    C c(1);
    C::iterator i = c.end();
    T j = *i;
    assert(false);
    ((void)j);
}
