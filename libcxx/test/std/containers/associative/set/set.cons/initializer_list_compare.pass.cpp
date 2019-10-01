//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03

// <set>

// class set

// set(initializer_list<value_type> il, const key_compare& comp = key_compare());

#include <set>
#include <cassert>
#include "test_macros.h"
#include "../../../test_compare.h"

int main(int, char**)
{
    typedef test_compare<std::less<int> > Cmp;
    typedef std::set<int, Cmp> C;
    typedef C::value_type V;
    C m({1, 2, 3, 4, 5, 6}, Cmp(10));
    assert(m.size() == 6);
    assert(distance(m.begin(), m.end()) == 6);
    C::const_iterator i = m.cbegin();
    assert(*i == V(1));
    assert(*++i == V(2));
    assert(*++i == V(3));
    assert(*++i == V(4));
    assert(*++i == V(5));
    assert(*++i == V(6));
    assert(m.key_comp() == Cmp(10));

  return 0;
}
