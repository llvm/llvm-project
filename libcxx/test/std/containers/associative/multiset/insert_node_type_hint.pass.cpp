//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// <set>

// class multiset

// iterator insert(const_iterator hint, node_type&&);

#include <set>
#include "test_macros.h"
#include "min_allocator.h"

template <class Container>
typename Container::node_type
node_factory(typename Container::key_type const& key)
{
    static Container c;
    auto it = c.insert(key);
    return c.extract(it);
}

template <class Container>
void test(Container& c)
{
    auto* nf = &node_factory<Container>;

    for (int i = 0; i != 10; ++i)
    {
        typename Container::node_type node = nf(i);
        assert(!node.empty());
        std::size_t prev = c.size();
        auto it = c.insert(c.end(), std::move(node));
        assert(prev + 1 == c.size());
        assert(*it == i);
    }

    assert(c.size() == 10);

    for (int i = 0; i != 10; ++i)
    {
        assert(c.count(i) == 1);
    }
}

int main(int, char**)
{
    std::multiset<int> m;
    test(m);
    std::multiset<int, std::less<int>, min_allocator<int>> m2;
    test(m2);

  return 0;
}
