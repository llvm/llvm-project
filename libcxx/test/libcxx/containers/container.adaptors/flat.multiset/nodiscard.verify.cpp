//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23

// <flat_set>

// Check that functions are marked [[nodiscard]]

#include <flat_set>
#include <utility>

template <typename T>
struct TransparentKey {
  T t;

  constexpr explicit operator T() const { return t; }
};

struct TransparentCompare {
  using is_transparent = void; // This makes the comparator transparent

  template <typename T>
  constexpr bool operator()(const T& t, const TransparentKey<T>& transparent) const {
    return t < transparent.t;
  }

  template <typename T>
  constexpr bool operator()(const TransparentKey<T>& transparent, const T& t) const {
    return transparent.t < t;
  }

  template <typename T>
  constexpr bool operator()(const T& t1, const T& t2) const {
    return t1 < t2;
  }
};

void test() {
  std::flat_multiset<int, TransparentCompare> fm;
  const std::flat_multiset<int, TransparentCompare> cfm;

  fm.begin();  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  cfm.begin(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  fm.end();    // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  cfm.end();   // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  fm.rbegin();  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  cfm.rbegin(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  fm.rend();    // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  cfm.rend();   // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  cfm.cbegin();  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  cfm.cend();    // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  cfm.crbegin(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  cfm.crend();   // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  fm.empty();    // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  fm.size();     // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  fm.max_size(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  std::move(fm).extract(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  fm.value_comp(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  fm.key_comp();   // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  int key = 0;
  TransparentKey<int> tkey;

  fm.find(key);   // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  cfm.find(key);  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  fm.find(tkey);  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  cfm.find(tkey); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  fm.count(key);  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  fm.count(tkey); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  fm.contains(key);  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  fm.contains(tkey); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  fm.lower_bound(key);   // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  cfm.lower_bound(key);  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  fm.lower_bound(tkey);  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  cfm.lower_bound(tkey); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  fm.upper_bound(key);   // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  cfm.upper_bound(key);  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  fm.upper_bound(tkey);  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  cfm.upper_bound(tkey); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  fm.equal_range(key);   // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  cfm.equal_range(key);  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  fm.equal_range(tkey);  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  cfm.equal_range(tkey); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
}
