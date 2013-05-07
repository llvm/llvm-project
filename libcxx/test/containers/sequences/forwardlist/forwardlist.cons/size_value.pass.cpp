//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <forward_list>

// forward_list(size_type n, const value_type& v);

#include <forward_list>
#include <cassert>

int main()
{
    {
        typedef int T;
        typedef std::forward_list<T> C;
        T v(6);
        unsigned N = 10;
        C c(N, v);
        unsigned n = 0;
        for (C::const_iterator i = c.begin(), e = c.end(); i != e; ++i, ++n)
            assert(*i == v);
        assert(n == N);
    }
}
