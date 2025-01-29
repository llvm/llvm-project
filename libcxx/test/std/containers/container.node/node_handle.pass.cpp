//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

#include <unordered_set>
#include <unordered_map>
#include <set>
#include <map>
#include "test_macros.h"
#include "min_allocator.h"

// [container.node.overview] Table 83.
template <class K, class T, class C1, class C2, class H1, class H2, class E1, class E2, class A_set, class A_map>
struct node_compatibility_table
{
    static constexpr bool value =
        std::is_same_v<typename std::map<K, T, C1, A_map>::node_type,               typename std::map<K, T, C2, A_map>::node_type> &&
        std::is_same_v<typename std::map<K, T, C1, A_map>::node_type,               typename std::multimap<K, T, C2, A_map>::node_type> &&
        std::is_same_v<typename std::set<K, C1, A_set>::node_type,                  typename std::set<K, C2, A_set>::node_type> &&
        std::is_same_v<typename std::set<K, C1, A_set>::node_type,                  typename std::multiset<K, C2, A_set>::node_type> &&
        std::is_same_v<typename std::unordered_map<K, T, H1, E1, A_map>::node_type, typename std::unordered_map<K, T, H2, E2, A_map>::node_type> &&
        std::is_same_v<typename std::unordered_map<K, T, H1, E1, A_map>::node_type, typename std::unordered_multimap<K, T, H2, E2, A_map>::node_type> &&
        std::is_same_v<typename std::unordered_set<K, H1, E1, A_set>::node_type,    typename std::unordered_set<K, H2, E2, A_set>::node_type> &&
        std::is_same_v<typename std::unordered_set<K, H1, E1, A_set>::node_type,    typename std::unordered_multiset<K, H2, E2, A_set>::node_type>;
};

template <class T> struct my_hash
{
    using argument_type = T;
    using result_type = std::size_t;
    my_hash() = default;
    std::size_t operator()(const T&) const {return 0;}
};

template <class T> struct my_compare
{
    my_compare() = default;
    bool operator()(const T&, const T&) const {return true;}
};

template <class T> struct my_equal
{
    my_equal() = default;
    bool operator()(const T&, const T&) const {return true;}
};

struct Static
{
    Static() = default;
    Static(const Static&) = delete;
    Static(Static&&) = delete;
    Static& operator=(const Static&) = delete;
    Static& operator=(Static&&) = delete;
};

template <>
struct std::hash<Static> {
  using argument_type = Static;
  using result_type   = std::size_t;
  hash()              = default;
  std::size_t operator()(const Static&) const;
};

static_assert(node_compatibility_table<
                  int, int, std::less<int>, std::less<int>, std::hash<int>,
                  std::hash<int>, std::equal_to<int>, std::equal_to<int>,
                  std::allocator<int>,
                  std::allocator<std::pair<const int, int>>>::value,
              "");

static_assert(
    node_compatibility_table<int, int, std::less<int>, my_compare<int>,
                             std::hash<int>, my_hash<int>, std::equal_to<int>,
                             my_equal<int>, std::allocator<int>,
                             std::allocator<std::pair<const int, int>>>::value,
    "");

static_assert(node_compatibility_table<
                  Static, int, my_compare<Static>, std::less<Static>,
                  my_hash<Static>, std::hash<Static>, my_equal<Static>,
                  std::equal_to<Static>, min_allocator<Static>,
                  min_allocator<std::pair<const Static, int>>>::value,
              "");

template <class Container>
void test_node_handle_operations()
{
    Container c;

    typename Container::node_type nt1, nt2 = c.extract(c.emplace().first);
    assert(nt2.get_allocator() == c.get_allocator());
    assert(!nt2.empty());
    assert(nt1.empty());
    std::swap(nt1, nt2);
    assert(nt1.get_allocator() == c.get_allocator());
    assert(nt2.empty());
}

template <class Container>
void test_node_handle_operations_multi()
{
    Container c;

    typename Container::node_type nt1, nt2 = c.extract(c.emplace());
    assert(nt2.get_allocator() == c.get_allocator());
    assert(!nt2.empty());
    assert(nt1.empty());
    std::swap(nt1, nt2);
    assert(nt1.get_allocator() == c.get_allocator());
    assert(nt2.empty());
}

template <class> void test_typedef() {}

template <class Container>
void test_insert_return_type()
{
    test_typedef<typename Container::insert_return_type>();
}

int main(int, char**)
{
    test_node_handle_operations<std::map<int, int>>();
    test_node_handle_operations_multi<std::multimap<int, int>>();
    test_node_handle_operations<std::set<int>>();
    test_node_handle_operations_multi<std::multiset<int>>();
    test_node_handle_operations<std::unordered_map<int, int>>();
    test_node_handle_operations_multi<std::unordered_multimap<int, int>>();
    test_node_handle_operations<std::unordered_set<int>>();
    test_node_handle_operations_multi<std::unordered_multiset<int>>();

    test_insert_return_type<std::map<int, int>>();
    test_insert_return_type<std::set<int>>();
    test_insert_return_type<std::unordered_map<int, int>>();
    test_insert_return_type<std::unordered_set<int>>();

  return 0;
}
