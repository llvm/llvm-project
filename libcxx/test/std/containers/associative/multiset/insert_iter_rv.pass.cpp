//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// <set>

// class multiset

// iterator insert(const_iterator position, value_type&& v);

#include <set>
#include <cassert>

#include "test_macros.h"
#include "MoveOnly.h"
#include "min_allocator.h"

int main(int, char**)
{
    {
        typedef std::multiset<MoveOnly> M;
        typedef M::iterator R;
        M m;
        R r = m.insert(m.cend(), M::value_type(2));
        assert(r == m.begin());
        assert(m.size() == 1);
        assert(*r == 2);

        r = m.insert(m.cend(), M::value_type(1));
        assert(r == m.begin());
        assert(m.size() == 2);
        assert(*r == 1);

        r = m.insert(m.cend(), M::value_type(3));
        assert(r == std::prev(m.end()));
        assert(m.size() == 3);
        assert(*r == 3);

        r = m.insert(m.cend(), M::value_type(3));
        assert(r == std::prev(m.end()));
        assert(m.size() == 4);
        assert(*r == 3);
    }
    {
        typedef std::multiset<MoveOnly, std::less<MoveOnly>, min_allocator<MoveOnly>> M;
        typedef M::iterator R;
        M m;
        R r = m.insert(m.cend(), M::value_type(2));
        assert(r == m.begin());
        assert(m.size() == 1);
        assert(*r == 2);

        r = m.insert(m.cend(), M::value_type(1));
        assert(r == m.begin());
        assert(m.size() == 2);
        assert(*r == 1);

        r = m.insert(m.cend(), M::value_type(3));
        assert(r == std::prev(m.end()));
        assert(m.size() == 3);
        assert(*r == 3);

        r = m.insert(m.cend(), M::value_type(3));
        assert(r == std::prev(m.end()));
        assert(m.size() == 4);
        assert(*r == 3);
    }

  return 0;
}
