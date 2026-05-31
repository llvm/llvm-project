//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <flat_set>

// Comprehensive exercise of every constructor and member function of
// std::flat_multiset<bool>.

#include <algorithm>
#include <cassert>
#include <flat_set>
#include <functional>
#include <initializer_list>
#include <ranges>
#include <utility>
#include <vector>

#include "test_macros.h"

using S = std::flat_multiset<bool>;

template <class S_>
void check_sorted(const S_& s) {
  auto it = s.begin();
  if (it == s.end())
    return;
  auto prev = it;
  for (++it; it != s.end(); ++it) {
    assert(!(*it < *prev));
    prev = it;
  }
}

void test_constructors() {
  static_assert(std::is_same_v<S::value_type, bool>);
  static_assert(std::is_same_v<S::key_type, bool>);

  // (1) default
  {
    S s;
    assert(s.empty());
  }

  // (2) explicit Compare
  {
    S s{S::key_compare{}};
    assert(s.empty());
  }

  // (3) from a container — duplicates kept
  {
    S::container_type cont{true, false, true, false};
    S s{std::move(cont)};
    assert(s.size() == 4);
    check_sorted(s);
  }

  // (4) sorted_equivalent + container
  {
    S::container_type cont{false, false, true};
    S s{std::sorted_equivalent, std::move(cont)};
    assert(s.size() == 3);
  }

  // (5) iterator range — duplicates kept
  {
    bool ar[] = {true, false, true, false};
    S s(ar, ar + 4);
    assert(s.size() == 4);
    check_sorted(s);
  }

  // (6) sorted_equivalent iterator range
  {
    bool ar[] = {false, false, true};
    S s(std::sorted_equivalent, ar, ar + 3);
    assert(s.size() == 3);
  }

  // (7) initializer_list
  {
    S s{true, false, true, false};
    assert(s.size() == 4);
    check_sorted(s);
  }

  // (8) sorted_equivalent + initializer_list
  {
    S s{std::sorted_equivalent, {false, false, true}};
    assert(s.size() == 3);
  }

  // (9) from_range
  {
    std::vector<bool> src{true, false, true};
    S s(std::from_range, src);
    assert(s.size() == 3);
  }

  // (10) copy
  {
    S a{true, false, true};
    S b{a};
    assert(std::ranges::equal(a, b));
  }

  // (11) move
  {
    S a{true, false, true};
    S b{std::move(a)};
    assert(b.size() == 3);
  }
}

void test_assignment() {
  // copy
  {
    S a{true, false, true};
    S b;
    b = a;
    assert(std::ranges::equal(a, b));
  }

  // move
  {
    S a{true, false, true};
    S b;
    b = std::move(a);
    assert(b.size() == 3);
  }

  // initializer_list
  {
    S a;
    a = {true, false, true, false};
    assert(a.size() == 4);
    check_sorted(a);
  }
}

void test_iterators_and_capacity() {
  S s{true, false, true};
  const S& cs = s;

  assert(!s.empty());
  assert(s.size() == 3);
  assert(s.max_size() > 0);

  assert(std::distance(s.begin(), s.end()) == 3);
  assert(std::distance(cs.begin(), cs.end()) == 3);
  assert(std::distance(s.cbegin(), s.cend()) == 3);

  assert(std::distance(s.rbegin(), s.rend()) == 3);
  assert(std::distance(cs.rbegin(), cs.rend()) == 3);
  assert(std::distance(s.crbegin(), s.crend()) == 3);

  S e;
  assert(e.empty());
  assert(e.begin() == e.end());
}

void test_modifiers() {
  // emplace — always succeeds (returns iterator)
  {
    S s;
    auto it = s.emplace(true);
    assert(*it == true);
    auto it2 = s.emplace(true); // duplicate allowed
    assert(*it2 == true);
    assert(s.size() == 2);
  }

  // emplace_hint
  {
    S s;
    auto it = s.emplace_hint(s.end(), false);
    assert(*it == false);
  }

  // insert(const value_type&)
  {
    S s;
    bool v  = true;
    auto it = s.insert(v);
    assert(*it == true);
    s.insert(v);
    assert(s.size() == 2);
  }

  // insert(value_type&&)
  {
    S s;
    auto it = s.insert(true);
    assert(*it == true);
  }

  // insert(const_iterator, const value_type&)
  {
    S s;
    bool v  = true;
    auto it = s.insert(s.end(), v);
    assert(*it == true);
  }

  // insert(const_iterator, value_type&&)
  {
    S s;
    auto it = s.insert(s.end(), false);
    assert(*it == false);
  }

  // insert(InputIterator, InputIterator)
  {
    S s;
    bool ar[] = {true, false, true};
    s.insert(ar, ar + 3);
    assert(s.size() == 3);
  }

  // insert(sorted_equivalent, InputIterator, InputIterator)
  {
    S s;
    bool ar[] = {false, true};
    s.insert(std::sorted_equivalent, ar, ar + 2);
    assert(s.size() == 2);
  }

  // insert(initializer_list)
  {
    S s;
    s.insert({true, false, true});
    assert(s.size() == 3);
  }

  // insert(sorted_equivalent, initializer_list)
  {
    S s;
    s.insert(std::sorted_equivalent, {false, true});
    assert(s.size() == 2);
  }

  // insert_range
  {
    S s;
    std::vector<bool> src{true, false, true};
    s.insert_range(src);
    assert(s.size() == 3);
  }

  // insert_range(sorted_equivalent, R)
  {
    S s;
    std::vector<bool> src{false, true};
    s.insert_range(std::sorted_equivalent, src);
    assert(s.size() == 2);
  }

  // erase(iterator)
  {
    S s{true, false};
    auto it = s.erase(s.begin());
    assert(s.size() == 1);
    (void)it;
  }

  // erase(const_iterator)
  {
    S s{true, false};
    auto it = s.erase(s.cbegin());
    assert(s.size() == 1);
    (void)it;
  }

  // erase(const_iterator, const_iterator)
  {
    S s{true, false};
    s.erase(s.cbegin(), s.cend());
    assert(s.empty());
  }

  // erase(const key_type&) — multiset: removes ALL matching
  {
    S s{true, false, true};
    auto n = s.erase(true);
    assert(n == 2);
    assert(s.size() == 1);
  }

  // extract / replace
  {
    S s{true, false};
    auto cont = std::move(s).extract();
    assert(s.empty());
    S s2;
    s2.replace(std::move(cont));
    assert(s2.size() == 2);
  }

  // clear
  {
    S s{true, false};
    s.clear();
    assert(s.empty());
  }

  // swap
  {
    S a{true};
    S b{false};
    a.swap(b);
    assert(*a.begin() == false);
    assert(*b.begin() == true);
  }

  // non-member swap
  {
    S a{true};
    S b{false};
    using std::swap;
    swap(a, b);
    assert(*a.begin() == false);
  }
}

void test_observers() {
  S s;
  auto kc = s.key_comp();
  auto vc = s.value_comp();
  assert(kc(false, true));
  assert(vc(false, true));
}

void test_operations() {
  S s{true, false, true};
  const S& cs = s;

  assert(s.find(true) != s.end());
  assert(cs.find(false) != cs.end());

  assert(s.count(true) == 2);
  assert(cs.count(false) == 1);

  assert(s.contains(true));
  assert(cs.contains(false));

  assert(s.lower_bound(false) == s.begin());
  assert(cs.lower_bound(false) == cs.begin());
  (void)s.upper_bound(true);
  (void)cs.upper_bound(true);

  auto er  = s.equal_range(true);
  auto cer = cs.equal_range(true);
  assert(std::distance(er.first, er.second) == 2);
  assert(std::distance(cer.first, cer.second) == 2);
}

void test_compare() {
  S a{true, false, true};
  S b{true, false, true};
  S c{true};

  assert(a == b);
  assert(!(a != b));
  assert(a != c);

  assert((a <=> b) == 0);
  assert((c <=> a) != 0);
}

void test_erase_if() {
  S s{true, false, true};
  auto erased = std::erase_if(s, [](bool x) { return x == true; });
  assert(erased == 2);
  assert(s.size() == 1);
  assert(!s.contains(true));
}

int main(int, char**) {
  test_constructors();
  test_assignment();
  test_iterators_and_capacity();
  test_modifiers();
  test_observers();
  test_operations();
  test_compare();
  test_erase_if();
  return 0;
}
