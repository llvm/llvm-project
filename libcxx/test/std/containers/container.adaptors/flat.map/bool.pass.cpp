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
// std::flat_map with `bool` used as the key and/or mapped type.

#include <algorithm>
#include <cassert>
#include <deque>
#include <flat_map>
#include <functional>
#include <initializer_list>
#include <ranges>
#include <stdexcept>
#include <utility>
#include <vector>

#include "test_macros.h"

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

template <class M>
void check_sorted_unique(const M& m) {
  auto it = m.begin();
  if (it == m.end())
    return;
  auto prev = it;
  for (++it; it != m.end(); ++it) {
    assert(prev->first < it->first);
    prev = it;
  }
}

// ---------------------------------------------------------------------------
// Constructors
// ---------------------------------------------------------------------------

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
    assert(m.size() == 0);
  }

  // (2) explicit Compare
  {
    M m{typename M::key_compare{}};
    assert(m.empty());
  }

  // (3) from two containers
  {
    typename M::key_container_type keys{k0, k1};
    typename M::mapped_container_type vals{v0, v1};
    M m{std::move(keys), std::move(vals)};
    assert(m.size() == 2);
    check_sorted_unique(m);
  }

  // (4) sorted_unique + two containers
  {
    typename M::key_container_type keys{k0, k1};
    typename M::mapped_container_type vals{v0, v1};
    // Ensure sorted: k0 < k1.
    if (!(k0 < k1))
      std::swap(keys.front(), keys.back()), std::swap(vals.front(), vals.back());
    M m{std::sorted_unique, std::move(keys), std::move(vals)};
    assert(m.size() == 2);
    check_sorted_unique(m);
  }

  // (5) iterator range, unsorted with duplicates
  {
    value_type ar[] = {value_type{k1, v1}, value_type{k0, v0}, value_type{k0, v1}};
    M m{ar, ar + 3};
    assert(m.size() == 2); // duplicates collapsed
    check_sorted_unique(m);
  }

  // (6) sorted_unique iterator range
  {
    value_type ar[2];
    if (k0 < k1) {
      ar[0] = value_type{k0, v0};
      ar[1] = value_type{k1, v1};
    } else {
      ar[0] = value_type{k1, v1};
      ar[1] = value_type{k0, v0};
    }
    M m{std::sorted_unique, ar, ar + 2};
    assert(m.size() == 2);
    check_sorted_unique(m);
  }

  // (7) initializer_list
  {
    M m{{value_type{k1, v1}, value_type{k0, v0}}};
    assert(m.size() == 2);
    check_sorted_unique(m);
  }

  // (8) sorted_unique + initializer_list
  {
    if (k0 < k1) {
      M m{std::sorted_unique, {value_type{k0, v0}, value_type{k1, v1}}};
      assert(m.size() == 2);
      check_sorted_unique(m);
    }
  }

  // (9) from_range (C++23 range-based)
  {
    std::vector<value_type> src{value_type{k0, v0}, value_type{k1, v1}};
    M m(std::from_range, src);
    assert(m.size() == 2);
    check_sorted_unique(m);
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

// ---------------------------------------------------------------------------
// Assignment
// ---------------------------------------------------------------------------

template <class M, class K, class V>
void test_assignment(K k0, K k1, V v0, V v1) {
  using value_type = typename M::value_type;

  // copy assignment
  {
    M a{{value_type{k0, v0}, value_type{k1, v1}}};
    M b;
    b = a;
    assert(b.size() == 2);
    assert(std::ranges::equal(a, b));
  }

  // move assignment
  {
    M a{{value_type{k0, v0}, value_type{k1, v1}}};
    M b;
    b = std::move(a);
    assert(b.size() == 2);
  }

  // initializer_list assignment
  {
    M a;
    a = {value_type{k1, v1}, value_type{k0, v0}};
    assert(a.size() == 2);
    check_sorted_unique(a);
  }
}

// ---------------------------------------------------------------------------
// Iterators, capacity
// ---------------------------------------------------------------------------

template <class M, class K, class V>
void test_iterators_and_capacity(K k0, K k1, V v0, V v1) {
  using value_type = typename M::value_type;

  M m{{value_type{k0, v0}, value_type{k1, v1}}};
  const M& cm = m;

  // capacity
  assert(!m.empty());
  assert(m.size() == 2);
  assert(m.max_size() > 0);

  // iterators (forward)
  assert(std::distance(m.begin(), m.end()) == 2);
  assert(std::distance(cm.begin(), cm.end()) == 2);
  assert(std::distance(m.cbegin(), m.cend()) == 2);

  // iterators (reverse)
  assert(std::distance(m.rbegin(), m.rend()) == 2);
  assert(std::distance(cm.rbegin(), cm.rend()) == 2);
  assert(std::distance(m.crbegin(), m.crend()) == 2);

  // empty()
  M e;
  assert(e.empty());
  assert(e.begin() == e.end());
}

// ---------------------------------------------------------------------------
// Element access (operator[], at) — only meaningful for flat_map
// ---------------------------------------------------------------------------

template <class M, class K, class V>
void test_access(K k0, K k1, V v0, V v1) {
  using value_type = typename M::value_type;

  M m{{value_type{k0, v0}, value_type{k1, v1}}};

  // operator[] (const K&)
  {
    const K kk = k0;
    auto&& r   = m[kk];
    assert(static_cast<V>(r) == v0);
  }

  // operator[] (K&&)
  {
    K kk     = k1;
    auto&& r = m[std::move(kk)];
    assert(static_cast<V>(r) == v1);
  }

  // operator[] inserts default when missing — only test when K has a missing value.
  // For bool keys this is degenerate (only two possible keys); skipped.

  // at (const K&) — present
  {
    const K kk = k0;
    auto&& r   = m.at(kk);
    assert(static_cast<V>(r) == v0);
  }
  // at (const K&) const — present
  {
    const M& cm = m;
    auto&& r    = cm.at(k1);
    assert(static_cast<V>(r) == v1);
  }
  // at — missing: only testable with int keys; bool keys are both present once the
  // map has two entries, so at(missing) throws only for K=int with a non-{0,1} value.
}

template <class M, class K, class V>
void test_access_missing(K /*present*/, K missing) {
  M m;
  try {
    (void)m.at(missing);
    assert(false && "at(missing) on empty map should throw");
  } catch (const std::out_of_range&) {
    // expected
  }
}

// ---------------------------------------------------------------------------
// Modifiers
// ---------------------------------------------------------------------------

template <class M, class K, class V>
void test_modifiers(K k0, K k1, V v0, V v1) {
  using value_type = typename M::value_type;

  // emplace — new
  {
    M m;
    auto r = m.emplace(k0, v0);
    assert(r.second);
    assert(r.first->first == k0);
    assert(m.size() == 1);
  }

  // emplace — duplicate
  {
    M m{{value_type{k0, v0}}};
    auto r = m.emplace(k0, v1);
    assert(!r.second);
    assert(m.size() == 1);
  }

  // emplace_hint
  {
    M m;
    auto it = m.emplace_hint(m.end(), k0, v0);
    assert(it->first == k0);
    assert(m.size() == 1);
  }

  // insert(const value_type&)
  {
    M m;
    value_type p{k0, v0};
    auto r = m.insert(p);
    assert(r.second && m.size() == 1);
  }

  // insert(value_type&&)
  {
    M m;
    auto r = m.insert(value_type{k1, v1});
    assert(r.second && m.size() == 1);
  }

  // insert(const_iterator, const value_type&)
  {
    M m;
    value_type p{k0, v0};
    auto it = m.insert(m.end(), p);
    assert(it->first == k0 && m.size() == 1);
  }

  // insert(const_iterator, value_type&&)
  {
    M m;
    auto it = m.insert(m.end(), value_type{k1, v1});
    assert(it->first == k1 && m.size() == 1);
  }

  // insert(InputIterator, InputIterator)
  {
    M m;
    value_type ar[] = {value_type{k1, v1}, value_type{k0, v0}};
    m.insert(ar, ar + 2);
    assert(m.size() == 2);
    check_sorted_unique(m);
  }

  // insert(sorted_unique_t, InputIterator, InputIterator)
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
    m.insert(std::sorted_unique, ar, ar + 2);
    assert(m.size() == 2);
  }

  // insert(initializer_list)
  {
    M m;
    m.insert({value_type{k1, v1}, value_type{k0, v0}});
    assert(m.size() == 2);
    check_sorted_unique(m);
  }

  // insert(sorted_unique_t, initializer_list)
  if (k0 < k1) {
    M m;
    m.insert(std::sorted_unique, {value_type{k0, v0}, value_type{k1, v1}});
    assert(m.size() == 2);
  }

  // insert_range
  {
    M m;
    std::vector<value_type> src{value_type{k0, v0}, value_type{k1, v1}};
    m.insert_range(src);
    assert(m.size() == 2);
  }

  // insert_range(sorted_unique_t, R)
  if (k0 < k1) {
    M m;
    std::vector<value_type> src{value_type{k0, v0}, value_type{k1, v1}};
    m.insert_range(std::sorted_unique, src);
    assert(m.size() == 2);
  }

  // insert_or_assign(const K&, V&&)
  {
    M m;
    auto r1 = m.insert_or_assign(k0, v0);
    assert(r1.second && m.size() == 1);
    auto r2 = m.insert_or_assign(k0, v1);
    assert(!r2.second);
    assert(static_cast<V>(m.at(k0)) == v1);
  }

  // insert_or_assign(K&&, V&&)
  {
    M m;
    K kk    = k1;
    auto r1 = m.insert_or_assign(std::move(kk), v1);
    assert(r1.second);
  }

  // insert_or_assign(const_iterator, const K&, V&&)
  {
    M m;
    auto it = m.insert_or_assign(m.end(), k0, v0);
    assert(it->first == k0);
  }

  // insert_or_assign(const_iterator, K&&, V&&)
  {
    M m;
    K kk    = k0;
    auto it = m.insert_or_assign(m.end(), std::move(kk), v0);
    assert(it->first == k0);
  }

  // try_emplace(const K&, args...)
  {
    M m;
    auto r1 = m.try_emplace(k0, v0);
    assert(r1.second);
    auto r2 = m.try_emplace(k0, v1);
    assert(!r2.second);
    assert(static_cast<V>(m.at(k0)) == v0); // value unchanged
  }

  // try_emplace(K&&, args...)
  {
    M m;
    K kk    = k1;
    auto r1 = m.try_emplace(std::move(kk), v1);
    assert(r1.second);
  }

  // try_emplace(const_iterator, const K&, args...)
  {
    M m;
    auto it = m.try_emplace(m.end(), k0, v0);
    assert(it->first == k0);
  }

  // try_emplace(const_iterator, K&&, args...)
  {
    M m;
    K kk    = k0;
    auto it = m.try_emplace(m.end(), std::move(kk), v0);
    assert(it->first == k0);
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
    auto it = m.erase(m.cbegin(), m.cend());
    assert(m.empty());
    (void)it;
  }

  // erase(const key_type&)
  {
    M m{{value_type{k0, v0}, value_type{k1, v1}}};
    auto n = m.erase(k0);
    assert(n == 1 && m.size() == 1);
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

  // swap (member)
  {
    M a{{value_type{k0, v0}}};
    M b{{value_type{k1, v1}}};
    a.swap(b);
    assert(a.size() == 1 && b.size() == 1);
    assert(a.begin()->first == k1);
    assert(b.begin()->first == k0);
  }

  // swap (non-member)
  {
    M a{{value_type{k0, v0}}};
    M b{{value_type{k1, v1}}};
    using std::swap;
    swap(a, b);
    assert(a.begin()->first == k1);
    assert(b.begin()->first == k0);
  }
}

// ---------------------------------------------------------------------------
// Observers
// ---------------------------------------------------------------------------

template <class M, class K, class V>
void test_observers(K k0, K k1, V /*v0*/, V /*v1*/) {
  M m;
  auto kc = m.key_comp();
  auto vc = m.value_comp();
  assert(kc(k0, k1) == (k0 < k1));
  using value_type = typename M::value_type;
  value_type p0{k0, V{}};
  value_type p1{k1, V{}};
  assert(vc(p0, p1) == (k0 < k1));
}

// ---------------------------------------------------------------------------
// Operations: find/count/contains/lower_bound/upper_bound/equal_range
// ---------------------------------------------------------------------------

template <class M, class K, class V>
void test_operations(K k0, K k1, V v0, V v1) {
  using value_type = typename M::value_type;
  M m{{value_type{k0, v0}, value_type{k1, v1}}};
  const M& cm = m;

  // find
  assert(m.find(k0) != m.end());
  assert(cm.find(k0) != cm.end());

  // count
  assert(m.count(k0) == 1);
  assert(cm.count(k0) == 1);

  // contains
  assert(m.contains(k0));
  assert(cm.contains(k0));

  // lower_bound / upper_bound
  assert(m.lower_bound(k0) != m.end());
  assert(cm.lower_bound(k0) != cm.end());
  assert(m.upper_bound(k0) != m.end() || k0 > k1); // upper bound may be end if k0 is max
  (void)m.upper_bound(k0);
  (void)cm.upper_bound(k0);

  // equal_range
  auto er  = m.equal_range(k0);
  auto cer = cm.equal_range(k0);
  assert(std::distance(er.first, er.second) == 1);
  assert(std::distance(cer.first, cer.second) == 1);
}

// ---------------------------------------------------------------------------
// Comparison
// ---------------------------------------------------------------------------

template <class M, class K, class V>
void test_compare(K k0, K k1, V v0, V v1) {
  using value_type = typename M::value_type;
  M a{{value_type{k0, v0}, value_type{k1, v1}}};
  M b{{value_type{k0, v0}, value_type{k1, v1}}};
  M c{{value_type{k0, v0}}};

  assert(a == b);
  assert(!(a != b));
  assert(a != c);
  assert(!(a == c));

  // <=> synthesizes the relational operators
  assert((a <=> b) == 0);
  assert((c <=> a) != 0);
  assert((a < c) || (c < a) || (a == c));
}

// ---------------------------------------------------------------------------
// erase_if (non-member)
// ---------------------------------------------------------------------------

template <class M, class K, class V>
void test_erase_if(K k0, K k1, V v0, V v1) {
  using value_type = typename M::value_type;
  M m{{value_type{k0, v0}, value_type{k1, v1}}};

  auto erased = std::erase_if(m, [&](const auto& p) { return p.first == k0; });
  assert(erased == 1);
  assert(m.size() == 1);
  assert(!m.contains(k0));
}

// ---------------------------------------------------------------------------
// Drive: instantiate for the three bool combinations
// ---------------------------------------------------------------------------

void test_int_to_bool() {
  using M = std::flat_map<int, bool>;

  test_constructors<M>(1, 2, true, false);
  test_assignment<M>(1, 2, true, false);
  test_iterators_and_capacity<M>(1, 2, true, false);
  test_access<M>(1, 2, true, false);
  test_access_missing<M, int, bool>(/*present*/ 1, /*missing*/ 99);
  test_modifiers<M>(1, 2, true, false);
  test_observers<M>(1, 2, true, false);
  test_operations<M>(1, 2, true, false);
  test_compare<M>(1, 2, true, false);
  test_erase_if<M>(1, 2, true, false);
}

void test_bool_to_int() {
  using M = std::flat_map<bool, int>;

  test_constructors<M>(false, true, 10, 20);
  test_assignment<M>(false, true, 10, 20);
  test_iterators_and_capacity<M>(false, true, 10, 20);
  test_access<M>(false, true, 10, 20);

  // No "missing key" test for bool key: every flat_map<bool,V> has at most 2 entries
  // and once both are inserted no bool is missing. Skipped.

  test_modifiers<M>(false, true, 10, 20);
  test_observers<M>(false, true, 10, 20);
  test_operations<M>(false, true, 10, 20);
  test_compare<M>(false, true, 10, 20);
  test_erase_if<M>(false, true, 10, 20);
}

void test_bool_to_bool() {
  using M = std::flat_map< bool, bool>;

  test_constructors<M>(false, true, false, true);
  test_assignment<M>(false, true, false, true);
  test_iterators_and_capacity<M>(false, true, false, true);
  test_access<M>(false, true, false, true);
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
