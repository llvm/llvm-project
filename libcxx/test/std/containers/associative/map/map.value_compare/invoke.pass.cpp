//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <map>

// class value_compare

// bool operator()( const value_type& lhs, const value_type& rhs ) const;

#include <map>
#include <cassert>
#include <string>
#include <utility>

template <typename Map>
struct CallCompMember : Map::value_compare {
    CallCompMember(const typename Map::value_compare& vc) : Map::value_compare(vc) {}

    typedef typename Map::value_type value_type;
    bool operator()(const value_type& value1, const value_type& value2) const {
        return this->comp(value1.first, value2.first);
    }
};

int main(int, char**) {
    typedef std::map<int, std::string> map_type;

    map_type m;
    std::pair<map_type::iterator, bool> p1 = m.insert(map_type::value_type(1, "abc"));
    std::pair<map_type::iterator, bool> p2 = m.insert(map_type::value_type(2, "abc"));

    const map_type::value_compare vc = m.value_comp();
    CallCompMember<map_type> call_comp = m.value_comp();

    assert(vc(*p1.first, *p2.first));
    assert(call_comp(*p1.first, *p2.first));

    assert(!vc(*p2.first, *p1.first));
    assert(!call_comp(*p2.first, *p1.first));

    return 0;
}
