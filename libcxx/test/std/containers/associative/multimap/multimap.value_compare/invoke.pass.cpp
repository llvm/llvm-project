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

template <typename MMap>
struct CallCompMember : MMap::value_compare {
    CallCompMember(const typename MMap::value_compare& vc) : MMap::value_compare(vc) {}

    typedef typename MMap::value_type value_type;
    bool operator()(const value_type& value1, const value_type& value2) const {
        return this->comp(value1.first, value2.first);
    }
};

int main(int, char**) {
    typedef std::multimap<int, std::string> map_type;

    map_type m;
    map_type::iterator i1 = m.insert(map_type::value_type(1, "abc"));
    map_type::iterator i2 = m.insert(map_type::value_type(2, "abc"));

    const map_type::value_compare vc = m.value_comp();
    CallCompMember<map_type> call_comp = m.value_comp();

    assert(vc(*i1, *i2));
    assert(call_comp(*i1, *i2));

    assert(!vc(*i2, *i1));
    assert(!call_comp(*i2, *i1));

    return 0;
}
