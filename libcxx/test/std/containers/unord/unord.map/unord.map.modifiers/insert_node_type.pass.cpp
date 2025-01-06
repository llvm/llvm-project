//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// <unordered_map>

// class unordered_map

// insert_return_type insert(node_type&&);

#include <memory>
#include <unordered_map>
#include "test_macros.h"
#include "min_allocator.h"

template <class Container, class T>
void verify_insert_return_type(T&& t) {
  using verified_type = std::remove_cv_t<std::remove_reference_t<T>>;
  static_assert(std::is_aggregate_v<verified_type>);
  static_assert(std::is_same_v<verified_type, typename Container::insert_return_type>);

  auto& [pos, ins, nod] = t;

  static_assert(std::is_same_v<decltype(pos), typename Container::iterator>);
  static_assert(std::is_same_v<decltype(t.position), typename Container::iterator>);
  assert(std::addressof(pos) == std::addressof(t.position));

  static_assert(std::is_same_v<decltype(ins), bool>);
  static_assert(std::is_same_v<decltype(t.inserted), bool>);
  assert(&ins == &t.inserted);

  static_assert(std::is_same_v<decltype(nod), typename Container::node_type>);
  static_assert(std::is_same_v<decltype(t.node), typename Container::node_type>);
  assert(std::addressof(nod) == std::addressof(t.node));
}

template <class Container>
typename Container::node_type
node_factory(typename Container::key_type const& key,
             typename Container::mapped_type const& mapped)
{
    static Container c;
    auto p = c.insert({key, mapped});
    assert(p.second);
    return c.extract(p.first);
}

template <class Container>
void test(Container& c)
{
    auto* nf = &node_factory<Container>;

    for (int i = 0; i != 10; ++i)
    {
        typename Container::node_type node = nf(i, i + 1);
        assert(!node.empty());
        typename Container::insert_return_type irt = c.insert(std::move(node));
        assert(node.empty());
        assert(irt.inserted);
        assert(irt.node.empty());
        assert(irt.position->first == i && irt.position->second == i + 1);
        verify_insert_return_type<Container>(irt);
    }

    assert(c.size() == 10);

    { // Insert empty node.
        typename Container::node_type def;
        auto irt = c.insert(std::move(def));
        assert(def.empty());
        assert(!irt.inserted);
        assert(irt.node.empty());
        assert(irt.position == c.end());
        verify_insert_return_type<Container>(irt);
    }

    { // Insert duplicate node.
        typename Container::node_type dupl = nf(0, 42);
        auto irt = c.insert(std::move(dupl));
        assert(dupl.empty());
        assert(!irt.inserted);
        assert(!irt.node.empty());
        assert(irt.position == c.find(0));
        assert(irt.node.key() == 0 && irt.node.mapped() == 42);
        verify_insert_return_type<Container>(irt);
    }

    assert(c.size() == 10);

    for (int i = 0; i != 10; ++i)
    {
        assert(c.count(i) == 1);
        assert(c[i] == i + 1);
    }
}

int main(int, char**)
{
    std::unordered_map<int, int> m;
    test(m);
    std::unordered_map<int, int, std::hash<int>, std::equal_to<int>, min_allocator<std::pair<const int, int>>> m2;
    test(m2);

  return 0;
}
