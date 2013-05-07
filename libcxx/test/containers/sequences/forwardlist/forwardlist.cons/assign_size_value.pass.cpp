//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <forward_list>

// void assign(size_type n, const value_type& v);

#include <forward_list>
#include <cassert>
#include <iterator>

int main()
{
    {
        typedef int T;
        typedef std::forward_list<T> C;
        const T t1[] = {10, 11, 12, 13};
        C c(std::begin(t1), std::end(t1));
        c.assign(10, 1);
        int n = 0;
        for (C::const_iterator i = c.cbegin(); i != c.cend(); ++i, ++n)
            assert(*i == 1);
        assert(n == 10);
    }
    {
        typedef int T;
        typedef std::forward_list<T> C;
        const T t1[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
        C c(std::begin(t1), std::end(t1));
        c.assign(4, 10);
        int n = 0;
        for (C::const_iterator i = c.cbegin(); i != c.cend(); ++i, ++n)
            assert(*i == 10);
        assert(n == 4);
    }
}
