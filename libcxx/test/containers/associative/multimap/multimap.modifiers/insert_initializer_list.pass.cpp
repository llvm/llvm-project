//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <map>

// class multimap

// void insert(initializer_list<value_type> il);

#include <map>
#include <cassert>

int main()
{
#ifndef _LIBCPP_HAS_NO_GENERALIZED_INITIALIZERS
    typedef std::multimap<int, double> C;
    typedef C::value_type V;
    C m =
           {
               {1, 1},
               {1, 2},
               {2, 1},
               {2, 2},
               {3, 1},
               {3, 2}
           };
    m.insert(
               {
                   {1, 1.5},
                   {2, 1.5},
                   {3, 1.5},
               }
            );
    assert(m.size() == 9);
    assert(distance(m.begin(), m.end()) == 9);
    C::const_iterator i = m.cbegin();
    assert(*i == V(1, 1));
    assert(*++i == V(1, 2));
    assert(*++i == V(1, 1.5));
    assert(*++i == V(2, 1));
    assert(*++i == V(2, 2));
    assert(*++i == V(2, 1.5));
    assert(*++i == V(3, 1));
    assert(*++i == V(3, 2));
    assert(*++i == V(3, 1.5));
#endif  // _LIBCPP_HAS_NO_GENERALIZED_INITIALIZERS
}
