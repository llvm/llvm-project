//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <unordered_map>

// template <class Key, class T, class Hash = hash<Key>, class Pred = equal_to<Key>,
//           class Alloc = allocator<pair<const Key, T>>>
// class unordered_multimap

// void insert(initializer_list<value_type> il);

#include <unordered_map>
#include <string>
#include <cassert>

#include "test_iterators.h"

int main()
{
#ifndef _LIBCPP_HAS_NO_GENERALIZED_INITIALIZERS
    {
        typedef std::unordered_multimap<int, std::string> C;
        typedef std::pair<int, std::string> P;
        C c;
        c.insert(
                    {
                        P(1, "one"),
                        P(2, "two"),
                        P(3, "three"),
                        P(4, "four"),
                        P(1, "four"),
                        P(2, "four"),
                    }
                );
        assert(c.size() == 6);
        typedef std::pair<C::iterator, C::iterator> Eq;
        Eq eq = c.equal_range(1);
        assert(std::distance(eq.first, eq.second) == 2);
        C::iterator k = eq.first;
        assert(k->first == 1);
        assert(k->second == "one");
        ++k;
        assert(k->first == 1);
        assert(k->second == "four");
        eq = c.equal_range(2);
        assert(std::distance(eq.first, eq.second) == 2);
        k = eq.first;
        assert(k->first == 2);
        assert(k->second == "two");
        ++k;
        assert(k->first == 2);
        assert(k->second == "four");
        eq = c.equal_range(3);
        assert(std::distance(eq.first, eq.second) == 1);
        k = eq.first;
        assert(k->first == 3);
        assert(k->second == "three");
        eq = c.equal_range(4);
        assert(std::distance(eq.first, eq.second) == 1);
        k = eq.first;
        assert(k->first == 4);
        assert(k->second == "four");
        assert(std::distance(c.begin(), c.end()) == c.size());
        assert(std::distance(c.cbegin(), c.cend()) == c.size());
    }
#endif  // _LIBCPP_HAS_NO_GENERALIZED_INITIALIZERS
}
