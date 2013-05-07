//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <forward_list>

// forward_list(initializer_list<value_type> il);

#include <forward_list>
#include <cassert>

int main()
{
#ifndef _LIBCPP_HAS_NO_GENERALIZED_INITIALIZERS
    {
        typedef int T;
        typedef std::forward_list<T> C;
        C c = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
        unsigned n = 0;
        for (C::const_iterator i = c.begin(), e = c.end(); i != e; ++i, ++n)
            assert(*i == n);
        assert(n == 10);
    }
#endif  // _LIBCPP_HAS_NO_GENERALIZED_INITIALIZERS
}
