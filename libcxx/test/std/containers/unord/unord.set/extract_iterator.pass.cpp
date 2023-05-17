//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// <unordered_set>

// class unordered_set

// node_type extract(const_iterator);

#include <unordered_set>
#include "test_macros.h"
#include "min_allocator.h"
#include "Counter.h"

template <class Container>
void test(Container& c)
{
    std::size_t sz = c.size();

    for (auto first = c.cbegin(); first != c.cend();)
    {
        auto key_value = *first;
        typename Container::node_type t = c.extract(first++);
        --sz;
        assert(t.value() == key_value);
        assert(t.get_allocator() == c.get_allocator());
        assert(sz == c.size());
    }

    assert(c.size() == 0);
}

int main(int, char**)
{
    {
        using set_type = std::unordered_set<int>;
        set_type m = {1, 2, 3, 4, 5, 6};
        test(m);
    }

    {
        std::unordered_set<Counter<int>> m = {1, 2, 3, 4, 5, 6};
        assert(Counter_base::gConstructed == 6);
        test(m);
        assert(Counter_base::gConstructed == 0);
    }

    {
        using min_alloc_set = std::unordered_set<int, std::hash<int>, std::equal_to<int>, min_allocator<int>>;
        min_alloc_set m = {1, 2, 3, 4, 5, 6};
        test(m);
    }

  return 0;
}
