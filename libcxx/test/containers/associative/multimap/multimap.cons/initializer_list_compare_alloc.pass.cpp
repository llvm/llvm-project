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

// multimap(initializer_list<value_type> il, const key_compare& comp, const allocator_type& a);

#include <map>
#include <cassert>
#include "../../../test_compare.h"
#include "../../../test_allocator.h"

int main()
{
#ifndef _LIBCPP_HAS_NO_GENERALIZED_INITIALIZERS
    typedef test_compare<std::less<int> > Cmp;
    typedef test_allocator<std::pair<const int, double> > A;
    typedef std::multimap<int, double, Cmp, A> C;
    typedef C::value_type V;
    C m(
           {
               {1, 1},
               {1, 1.5},
               {1, 2},
               {2, 1},
               {2, 1.5},
               {2, 2},
               {3, 1},
               {3, 1.5},
               {3, 2}
           },
           Cmp(4), A(5)
        );
    assert(m.size() == 9);
    assert(distance(m.begin(), m.end()) == 9);
    C::const_iterator i = m.cbegin();
    assert(*i == V(1, 1));
    assert(*++i == V(1, 1.5));
    assert(*++i == V(1, 2));
    assert(*++i == V(2, 1));
    assert(*++i == V(2, 1.5));
    assert(*++i == V(2, 2));
    assert(*++i == V(3, 1));
    assert(*++i == V(3, 1.5));
    assert(*++i == V(3, 2));
    assert(m.key_comp() == Cmp(4));
    assert(m.get_allocator() == A(5));
#endif  // _LIBCPP_HAS_NO_GENERALIZED_INITIALIZERS
}
