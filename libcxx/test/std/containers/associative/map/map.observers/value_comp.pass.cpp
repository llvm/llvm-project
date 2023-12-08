//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <map>

// value_compare value_comp() const;

#include <map>
#include <cassert>
#include <string>

int main(int, char**) {
    typedef std::map<int, std::string> map_type;

    map_type m;
    std::pair<map_type::iterator, bool> p1 = m.insert(map_type::value_type(1, "abc"));
    std::pair<map_type::iterator, bool> p2 = m.insert(map_type::value_type(2, "abc"));

    const map_type& cm = m;

    assert(cm.value_comp()(*p1.first, *p2.first));
    assert(!cm.value_comp()(*p2.first, *p1.first));

    return 0;
}
