//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <flat_map>

// Comprehensive exercise of every constructor and member function of
// std::flat_multimap with `bool` used as the key and/or mapped type.

#include <algorithm>
#include <cassert>
#include <flat_map>
#include <functional>
#include <initializer_list>
#include <ranges>
#include <utility>
#include <vector>

#include "test_macros.h"

template <class M>
void check_sorted(const M& m) {
  auto it = m.begin();
  if (it == m.end())
    return;
  auto prev = it;
  for (++it; it != m.end(); ++it) {
    assert(!(it->first < prev->first));
    prev = it;
  }
}

template <class M, class K, class V>
void test_constructors(K k0, K k1, V v0, V v1) {
  using value_type = typename M::value_type;
  using key_type   = typename M::key_type;
  using mapped_t   = typename M::mapped_type;
  static_assert(std::is_same_v<key_type, K>);
  static_assert(std::is_same_v<mapped_t, V>);

  // (1) default
  {
    M m;
    assert(m.empty());
  }

  // (2) explicit Compare
  {
    M m{typename M::key_compare{}};
    assert(m.empty());
  }

  // (3) from two containers — duplicates are kept
  {
    typename M::key_container_type keys{k0, k1, k0};
    typename M::mapped_container_type vals{v0, v1, v1};
    M m{std::move(keys), std::move(vals)};
    assert(m.size() == 3);
    check_sorted(m);
  }

  // (4) sorted_equivalent + two containers
  {
    typename M::key_container_type keys{k0, k0, k1};
    typename M::mapped_container_type vals{v0, v1, v1};
    if (!(k0 < k1) && k0 != k1) {
      std::swap(keys.front(), keys.back());
      std::swap(vals.front(), vals.back());
    }
    M m{std::sorted_equivalent, std::move(keys), std::move(vals)};
    assert(m.size() == 3);
    check_sorted(m);
  }

  // (5) iterator range with duplicates — all kept
  {
    value_type ar[] = {value_type{k1, v1}, value_type{k0, v0}, value_type{k0, v1}};
    M m{ar, ar + 3};
    assert(m.size() == 3);
    check_sorted(m);
  }

  // (6) sorted_equivalent iterator range
  {
    value_type ar[3];
    if (k0 < k1) {
      ar[0] = value_type{k0, v0};
      ar[1] = value_type{k0, v1};
      ar[2] = value_type{k1, v1};
    } else {
      ar[0] = value_type{k1, v1};
      ar[1] = value_type{k0, v0};
      ar[2] = value_type{k0, v1};
    }
    M m{std::sorted_equivalent, ar, ar + 3};
    assert(m.size() == 3);
  }

  // (7) initializer_list
  {
    M m{{value_type{k1, v1}, value_type{k0, v0}, value_type{k0, v1}}};
    assert(m.size() == 3);
    check_sorted(m);
  }

  // (8) sorted_equivalent + initializer_list
  if (k0 < k1) {
    M m{std::sorted_equivalent, {value_type{k0, v0}, value_type{k0, v1}, value_type{k1, v1}}};
    assert(m.size() == 3);
  }

  // (9) from_range
  {
    std::vector<value_type> src{value_type{k0, v0}, value_type{k1, v1}, value_type{k0, v1}};
    M m(std::from_range, src);
    assert(m.size() == 3);
    check_sorted(m);
  }

  // (10) copy
  {
    M m{{value_type{k0, v0}, value_type{k1, v1}}};
    M c{m};
    assert(c.size() == m.size());
    assert(std::ranges::equal(m, c));
  }

  // (11) move
  {
    M m{{value_type{k0, v0}, value_type{k1, v1}}};
    M moved{std::move(m)};
    assert(moved.size() == 2);
  }
}

template <class M, class K, class V>
void test_assignment(K k0, K k1, V v0, V v1) {
  using value_type = typename M::value_type;

  // copy
  {
    M a{{value_type{k0, v0}, value_type{k1, v1}}};
    M b;
    b = a;
    assert(b.size() == 2);
    assert(std::ranges::equal(a, b));
  }

  // move
  {
    M a{{value_type{k0, v0}, value_type{k1, v1}}};
    M b;
    b = std::move(a);
    assert(b.size() == 2);
  }

  // initializer_list
  {
    M a;
    a = {value_type{k1, v1}, value_type{k0, v0}, value_type{k0, v1}};
    assert(a.size() == 3);
    check_sorted(a);
  }
}

template <class M, class K, class V>
void test_iterators_and_capacity(K k0, K k1, V v0, V v1) {
  using value_type = typename M::value_type;
  M m{{value_type{k0, v0}, value_type{k1, v1}, value_type{k0, v1}}};
  const M& cm = m;

  assert(!m.empty());
  assert(m.size() == 3);
  assert(m.max_size() > 0);

  assert(std::distance(m.begin(), m.end()) == 3);
  assert(std::distance(cm.begin(), cm.end()) == 3);
  assert(std::distance(m.cbegin(), m.cend()) == 3);

  assert(std::distance(m.rbegin(), m.rend()) == 3);
  assert(std::distance(cm.rbegin(), cm.rend()) == 3);
  assert(std::distance(m.crbegin(), m.crend()) == 3);

  M e;
  assert(e.empty());
  assert(e.begin() == e.end());
}

template <class M, class K, class V>
void test_modifiers(K k0, K k1, V v0, V v1) {
  using value_type = typename M::value_type;

  // emplace — always succeeds (returns iterator, not pair)
  {
    M m;
    auto it = m.emplace(k0, v0);
    assert(it->first == k0);
    assert(m.size() == 1);
    auto it2 = m.emplace(k0, v1); // duplicate allowed
    assert(it2->first == k0);
    assert(m.size() == 2);
  }

  // emplace_hint
  {
    M m;
    auto it = m.emplace_hint(m.end(), k0, v0);
    assert(it->first == k0);
  }

  // insert(const value_type&)
  {
    M m;
    value_type p{k0, v0};
    auto it = m.insert(p);
    assert(it->first == k0);
    auto it2 = m.insert(p);
    assert(m.size() == 2);
    (void)it2;
  }

  // insert(value_type&&)
  {
    M m;
    auto it = m.insert(value_type{k1, v1});
    assert(it->first == k1);
  }

  // insert(const_iterator, const value_type&)
  {
    M m;
    value_type p{k0, v0};
    auto it = m.insert(m.end(), p);
    assert(it->first == k0);
  }

  // insert(const_iterator, value_type&&)
  {
    M m;
    auto it = m.insert(m.end(), value_type{k1, v1});
    assert(it->first == k1);
  }

  // insert(InputIterator, InputIterator)
  {
    M m;
    value_type ar[] = {value_type{k1, v1}, value_type{k0, v0}, value_type{k0, v1}};
    m.insert(ar, ar + 3);
    assert(m.size() == 3);
    check_sorted(m);
  }

  // insert(sorted_equivalent, InputIterator, InputIterator)
  {
    M m;
    value_type ar[2];
    if (k0 < k1) {
      ar[0] = value_type{k0, v0};
      ar[1] = value_type{k1, v1};
    } else {
      ar[0] = value_type{k1, v1};
      ar[1] = value_type{k0, v0};
    }
    m.insert(std::sorted_equivalent, ar, ar + 2);
    assert(m.size() == 2);
  }

  // insert(initializer_list)
  {
    M m;
    m.insert({value_type{k1, v1}, value_type{k0, v0}});
    assert(m.size() == 2);
  }

  // insert(sorted_equivalent, initializer_list)
  if (k0 < k1) {
    M m;
    m.insert(std::sorted_equivalent, {value_type{k0, v0}, value_type{k1, v1}});
    assert(m.size() == 2);
  }

  // insert_range
  {
    M m;
    std::vector<value_type> src{value_type{k0, v0}, value_type{k1, v1}, value_type{k0, v1}};
    m.insert_range(src);
    assert(m.size() == 3);
  }

  // insert_range(sorted_equivalent, R)
  if (k0 < k1) {
    M m;
    std::vector<value_type> src{value_type{k0, v0}, value_type{k1, v1}};
    m.insert_range(std::sorted_equivalent, src);
    assert(m.size() == 2);
  }

  // erase(iterator)
  {
    M m{{value_type{k0, v0}, value_type{k1, v1}}};
    auto it = m.erase(m.begin());
    assert(m.size() == 1);
    (void)it;
  }

  // erase(const_iterator)
  {
    M m{{value_type{k0, v0}, value_type{k1, v1}}};
    auto it = m.erase(m.cbegin());
    assert(m.size() == 1);
    (void)it;
  }

  // erase(const_iterator, const_iterator)
  {
    M m{{value_type{k0, v0}, value_type{k1, v1}}};
    m.erase(m.cbegin(), m.cend());
    assert(m.empty());
  }

  // erase(const key_type&) — multimap: erases ALL matching
  {
    M m{{value_type{k0, v0}, value_type{k0, v1}, value_type{k1, v1}}};
    auto n = m.erase(k0);
    assert(n == 2);
    assert(m.size() == 1);
  }

  // extract / replace
  {
    M m{{value_type{k0, v0}, value_type{k1, v1}}};
    auto extracted = std::move(m).extract();
    assert(m.empty());
    M m2;
    m2.replace(std::move(extracted.keys), std::move(extracted.values));
    assert(m2.size() == 2);
  }

  // clear
  {
    M m{{value_type{k0, v0}, value_type{k1, v1}}};
    m.clear();
    assert(m.empty());
  }

  // swap
  {
    M a{{value_type{k0, v0}}};
    M b{{value_type{k1, v1}}};
    a.swap(b);
    assert(a.begin()->first == k1);
    assert(b.begin()->first == k0);
  }

  // non-member swap
  {
    M a{{value_type{k0, v0}}};
    M b{{value_type{k1, v1}}};
    using std::swap;
    swap(a, b);
    assert(a.begin()->first == k1);
    assert(b.begin()->first == k0);
  }
}

template <class M, class K, class V>
void test_observers(K k0, K k1, V, V) {
  M m;
  auto kc = m.key_comp();
  auto vc = m.value_comp();
  assert(kc(k0, k1) == (k0 < k1));
  using value_type = typename M::value_type;
  value_type p0{k0, V{}};
  value_type p1{k1, V{}};
  assert(vc(p0, p1) == (k0 < k1));
}

template <class M, class K, class V>
void test_operations(K k0, K k1, V v0, V v1) {
  using value_type = typename M::value_type;
  M m{{value_type{k0, v0}, value_type{k0, v1}, value_type{k1, v1}}};
  const M& cm = m;

  assert(m.find(k0) != m.end());
  assert(cm.find(k0) != cm.end());

  assert(m.count(k0) == 2);
  assert(cm.count(k0) == 2);
  assert(m.count(k1) == 1);

  assert(m.contains(k0));
  assert(cm.contains(k0));

  assert(m.lower_bound(k0) != m.end());
  assert(cm.lower_bound(k0) != cm.end());
  (void)m.upper_bound(k0);
  (void)cm.upper_bound(k0);

  auto er  = m.equal_range(k0);
  auto cer = cm.equal_range(k0);
  assert(std::distance(er.first, er.second) == 2);
  assert(std::distance(cer.first, cer.second) == 2);
}

template <class M, class K, class V>
void test_compare(K k0, K k1, V v0, V v1) {
  using value_type = typename M::value_type;
  M a{{value_type{k0, v0}, value_type{k1, v1}}};
  M b{{value_type{k0, v0}, value_type{k1, v1}}};
  M c{{value_type{k0, v0}}};

  assert(a == b);
  assert(!(a != b));
  assert(a != c);

  assert((a <=> b) == 0);
  assert((c <=> a) != 0);
}

template <class M, class K, class V>
void test_erase_if(K k0, K k1, V v0, V v1) {
  using value_type = typename M::value_type;
  M m{{value_type{k0, v0}, value_type{k0, v1}, value_type{k1, v1}}};

  auto erased = std::erase_if(m, [&](const auto& p) { return p.first == k0; });
  assert(erased == 2);
  assert(m.size() == 1);
  assert(!m.contains(k0));
}

void test_int_to_bool() {
  using M = std::flat_multimap<int, bool>;
  test_constructors<M>(1, 2, true, false);
  test_assignment<M>(1, 2, true, false);
  test_iterators_and_capacity<M>(1, 2, true, false);
  test_modifiers<M>(1, 2, true, false);
  test_observers<M>(1, 2, true, false);
  test_operations<M>(1, 2, true, false);
  test_compare<M>(1, 2, true, false);
  test_erase_if<M>(1, 2, true, false);
}

void test_bool_to_int() {
  using M = std::flat_multimap<bool, int>;
  test_constructors<M>(false, true, 10, 20);
  test_assignment<M>(false, true, 10, 20);
  test_iterators_and_capacity<M>(false, true, 10, 20);
  test_modifiers<M>(false, true, 10, 20);
  test_observers<M>(false, true, 10, 20);
  test_operations<M>(false, true, 10, 20);
  test_compare<M>(false, true, 10, 20);
  test_erase_if<M>(false, true, 10, 20);
}

void test_bool_to_bool() {
  using M = std::flat_multimap<bool, bool>;
  test_constructors<M>(false, true, false, true);
  test_assignment<M>(false, true, false, true);
  test_iterators_and_capacity<M>(false, true, false, true);
  test_modifiers<M>(false, true, false, true);
  test_observers<M>(false, true, false, true);
  test_operations<M>(false, true, false, true);
  test_compare<M>(false, true, false, true);
  test_erase_if<M>(false, true, false, true);
}

int main(int, char**) {
  test_int_to_bool();
  test_bool_to_int();
  test_bool_to_bool();
  return 0;
}
