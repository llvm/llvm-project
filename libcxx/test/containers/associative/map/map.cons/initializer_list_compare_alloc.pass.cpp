//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <map>

// class map

// map(initializer_list<value_type> il, const key_compare& comp, const allocator_type& a);

#include <map>
#include <cassert>
#include "../../../test_compare.h"
#include "../../../test_allocator.h"

int main()
{
#ifndef _LIBCPP_HAS_NO_GENERALIZED_INITIALIZERS
    typedef std::pair<const int, double> V;
    typedef test_compare<std::less<int> > C;
    typedef test_allocator<std::pair<const int, double> > A;
    std::map<int, double, C, A> m({
                                   {1, 1},
                                   {1, 1.5},
                                   {1, 2},
                                   {2, 1},
                                   {2, 1.5},
                                   {2, 2},
                                   {3, 1},
                                   {3, 1.5},
                                   {3, 2}
                                  }, C(3), A(6));
    assert(m.size() == 3);
    assert(distance(m.begin(), m.end()) == 3);
    assert(*m.begin() == V(1, 1));
    assert(*next(m.begin()) == V(2, 1));
    assert(*next(m.begin(), 2) == V(3, 1));
    assert(m.key_comp() == C(3));
    assert(m.get_allocator() == A(6));
#endif  // _LIBCPP_HAS_NO_GENERALIZED_INITIALIZERS
}
