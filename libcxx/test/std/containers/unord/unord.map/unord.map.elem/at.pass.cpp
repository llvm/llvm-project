//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// XFAIL: libcpp-no-exceptions
// <unordered_map>

// template <class Key, class T, class Hash = hash<Key>, class Pred = equal_to<Key>,
//           class Alloc = allocator<pair<const Key, T>>>
// class unordered_map

// mapped_type&       at(const key_type& k);
// const mapped_type& at(const key_type& k) const;

#include <unordered_map>
#include <string>
#include <cassert>

#include "MoveOnly.h"
#include "min_allocator.h"

int main()
{
    {
        typedef std::unordered_map<int, std::string> C;
        typedef std::pair<int, std::string> P;
        P a[] =
        {
            P(1, "one"),
            P(2, "two"),
            P(3, "three"),
            P(4, "four"),
            P(1, "four"),
            P(2, "four"),
        };
        C c(a, a + sizeof(a)/sizeof(a[0]));
        assert(c.size() == 4);
        c.at(1) = "ONE";
        assert(c.at(1) == "ONE");
        try
        {
            c.at(11) = "eleven";
            assert(false);
        }
        catch (std::out_of_range&)
        {
        }
        assert(c.size() == 4);
    }
    {
        typedef std::unordered_map<int, std::string> C;
        typedef std::pair<int, std::string> P;
        P a[] =
        {
            P(1, "one"),
            P(2, "two"),
            P(3, "three"),
            P(4, "four"),
            P(1, "four"),
            P(2, "four"),
        };
        const C c(a, a + sizeof(a)/sizeof(a[0]));
        assert(c.size() == 4);
        assert(c.at(1) == "one");
        try
        {
            c.at(11);
            assert(false);
        }
        catch (std::out_of_range&)
        {
        }
        assert(c.size() == 4);
    }
#if __cplusplus >= 201103L
    {
        typedef std::unordered_map<int, std::string, std::hash<int>, std::equal_to<int>,
                            min_allocator<std::pair<const int, std::string>>> C;
        typedef std::pair<int, std::string> P;
        P a[] =
        {
            P(1, "one"),
            P(2, "two"),
            P(3, "three"),
            P(4, "four"),
            P(1, "four"),
            P(2, "four"),
        };
        C c(a, a + sizeof(a)/sizeof(a[0]));
        assert(c.size() == 4);
        c.at(1) = "ONE";
        assert(c.at(1) == "ONE");
        try
        {
            c.at(11) = "eleven";
            assert(false);
        }
        catch (std::out_of_range&)
        {
        }
        assert(c.size() == 4);
    }
    {
        typedef std::unordered_map<int, std::string, std::hash<int>, std::equal_to<int>,
                            min_allocator<std::pair<const int, std::string>>> C;
        typedef std::pair<int, std::string> P;
        P a[] =
        {
            P(1, "one"),
            P(2, "two"),
            P(3, "three"),
            P(4, "four"),
            P(1, "four"),
            P(2, "four"),
        };
        const C c(a, a + sizeof(a)/sizeof(a[0]));
        assert(c.size() == 4);
        assert(c.at(1) == "one");
        try
        {
            c.at(11);
            assert(false);
        }
        catch (std::out_of_range&)
        {
        }
        assert(c.size() == 4);
    }
#endif
}
