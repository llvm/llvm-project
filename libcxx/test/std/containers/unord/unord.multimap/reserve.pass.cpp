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

// void rehash(size_type n);

#include <unordered_map>
#include <string>
#include <cassert>

#include "min_allocator.h"

template <class C>
void test(const C& c)
{
    assert(c.size() == 6);
    assert(c.find(1)->second == "one");
    assert(next(c.find(1))->second == "four");
    assert(c.find(2)->second == "two");
    assert(next(c.find(2))->second == "four");
    assert(c.find(3)->second == "three");
    assert(c.find(4)->second == "four");
}

void reserve_invariant(size_t n) // LWG #2156
{
    for (size_t i = 0; i < n; ++i)
    {
        std::unordered_multimap<size_t, size_t> c;
        c.reserve(n);
        size_t buckets = c.bucket_count();
        for (size_t j = 0; j < i; ++j)
        {
            c.insert(std::unordered_multimap<size_t, size_t>::value_type(i,i));
            assert(buckets == c.bucket_count());
        }
    }
}

int main()
{
    {
        typedef std::unordered_multimap<int, std::string> C;
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
        test(c);
        assert(c.bucket_count() >= 7);
        c.reserve(3);
        assert(c.bucket_count() == 7);
        test(c);
        c.max_load_factor(2);
        c.reserve(3);
        assert(c.bucket_count() == 3);
        test(c);
        c.reserve(31);
        assert(c.bucket_count() >= 16);
        test(c);
    }
#if __cplusplus >= 201103L
    {
        typedef std::unordered_multimap<int, std::string, std::hash<int>, std::equal_to<int>,
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
        test(c);
        assert(c.bucket_count() >= 7);
        c.reserve(3);
        assert(c.bucket_count() == 7);
        test(c);
        c.max_load_factor(2);
        c.reserve(3);
        assert(c.bucket_count() == 3);
        test(c);
        c.reserve(31);
        assert(c.bucket_count() >= 16);
        test(c);
    }
#endif
    reserve_invariant(20);
}
