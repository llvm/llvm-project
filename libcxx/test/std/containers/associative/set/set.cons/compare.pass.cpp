//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <set>

// class set

// explicit set(const value_compare& comp) const;
// value_compare and key_compare are the same type for set/multiset

// key_compare    key_comp() const;
// value_compare value_comp() const;

#include <set>
#include <cassert>

#include "../../../test_compare.h"

int main()
{
    typedef test_compare<std::less<int> > C;
    const std::set<int, C> m(C(3));
    assert(m.empty());
    assert(m.begin() == m.end());
    assert(m.key_comp() == C(3));
    assert(m.value_comp() == C(3));
}
