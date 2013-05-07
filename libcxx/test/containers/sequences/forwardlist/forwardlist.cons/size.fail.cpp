//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <forward_list>

// explicit forward_list(size_type n);

#include <forward_list>
#include <cassert>

#include "../../../DefaultOnly.h"

int main()
{
    {
        typedef DefaultOnly T;
        typedef std::forward_list<T> C;
        unsigned N = 10;
        C c = N;
        unsigned n = 0;
        for (C::const_iterator i = c.begin(), e = c.end(); i != e; ++i, ++n)
#ifndef _LIBCPP_HAS_NO_RVALUE_REFERENCES
            assert(*i == T());
#else
            ;
#endif
        assert(n == N);
    }
}
