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

// pair<const_iterator, const_iterator> equal_range(const key_type& k) const;

#include <unordered_map>
#include <string>
#include <cassert>

int main()
{
    {
        typedef std::unordered_multimap<int, std::string> C;
        typedef C::const_iterator I;
        typedef std::pair<int, std::string> P;
        P a[] =
        {
            P(10, "ten"),
            P(20, "twenty"),
            P(30, "thirty"),
            P(40, "fourty"),
            P(50, "fifty"),
            P(50, "fiftyA"),
            P(50, "fiftyB"),
            P(60, "sixty"),
            P(70, "seventy"),
            P(80, "eighty"),
        };
        const C c(std::begin(a), std::end(a));
        std::pair<I, I> r = c.equal_range(30);
        assert(std::distance(r.first, r.second) == 1);
        assert(r.first->first == 30);
        assert(r.first->second == "thirty");
        r = c.equal_range(5);
        assert(std::distance(r.first, r.second) == 0);
        r = c.equal_range(50);
        assert(r.first->first == 50);
        assert(r.first->second == "fifty");
        ++r.first;
        assert(r.first->first == 50);
        assert(r.first->second == "fiftyA");
        ++r.first;
        assert(r.first->first == 50);
        assert(r.first->second == "fiftyB");
    }
}
