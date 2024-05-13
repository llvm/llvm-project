//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <set>

// key_compare key_comp() const;
// value_compare value_comp() const;

#include <set>
#include <cassert>

int main(int, char**) {
    typedef std::set<int> set_type;

    set_type s;
    std::pair<set_type::iterator, bool> p1 = s.insert(1);
    std::pair<set_type::iterator, bool> p2 = s.insert(2);

    const set_type& cs = s;

    assert(cs.key_comp()(*p1.first, *p2.first));
    assert(!cs.key_comp()(*p2.first, *p1.first));

    assert(cs.value_comp()(*p1.first, *p2.first));
    assert(!cs.value_comp()(*p2.first, *p1.first));

    return 0;
}
