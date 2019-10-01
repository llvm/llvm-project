//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03, c++11, c++14

// <map>

// class map

// node_type extract(key_type const&);

#include <map>
#include "test_macros.h"
#include "min_allocator.h"
#include "Counter.h"

template <class Container, class KeyTypeIter>
void test(Container& c, KeyTypeIter first, KeyTypeIter last)
{
    size_t sz = c.size();
    assert((size_t)std::distance(first, last) == sz);

    for (KeyTypeIter copy = first; copy != last; ++copy)
    {
        typename Container::node_type t = c.extract(*copy);
        assert(!t.empty());
        --sz;
        assert(t.key() == *copy);
        t.key() = *first; // We should be able to mutate key.
        assert(t.key() == *first);
        assert(t.get_allocator() == c.get_allocator());
        assert(sz == c.size());
    }

    assert(c.size() == 0);

    for (KeyTypeIter copy = first; copy != last; ++copy)
    {
        typename Container::node_type t = c.extract(*copy);
        assert(t.empty());
    }
}

int main(int, char**)
{
    {
        std::map<int, int> m = {{1,1}, {2,2}, {3,3}, {4,4}, {5,5}, {6,6}};
        int keys[] = {1, 2, 3, 4, 5, 6};
        test(m, std::begin(keys), std::end(keys));
    }

    {
        std::map<Counter<int>, Counter<int>> m =
            {{1,1}, {2,2}, {3,3}, {4,4}, {5,5}, {6,6}};
        {
            Counter<int> keys[] = {1, 2, 3, 4, 5, 6};
            assert(Counter_base::gConstructed == 12+6);
            test(m, std::begin(keys), std::end(keys));
        }
        assert(Counter_base::gConstructed == 0);
    }

    {
        using min_alloc_map =
            std::map<int, int, std::less<int>,
                     min_allocator<std::pair<const int, int>>>;
        min_alloc_map m = {{1, 1}, {2, 2}, {3, 3}, {4, 4}, {5, 5}, {6, 6}};
        int keys[] = {1, 2, 3, 4, 5, 6};
        test(m, std::begin(keys), std::end(keys));
    }

  return 0;
}
