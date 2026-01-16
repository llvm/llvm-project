//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23

// <flat_map>

// Check that functions are marked [[nodiscard]]

#include <flat_map>
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
  std::flat_multimap<int, int, TransparentCompare> mm;
  const std::flat_multimap<int, int, TransparentCompare> cmm;

  mm.begin();  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  cmm.begin(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  mm.end();    // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  cmm.end();   // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  mm.rbegin();  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  cmm.rbegin(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  mm.rend();    // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  cmm.rend();   // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  cmm.cbegin();  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  cmm.cend();    // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  cmm.crbegin(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  cmm.crend();   // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  mm.empty();    // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  mm.size();     // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  mm.max_size(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  mm.key_comp();   // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  mm.value_comp(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  mm.keys();       // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  mm.values();     // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  int key = 0;
  TransparentKey<int> tkey;

  mm.find(key);   // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  cmm.find(key);  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  mm.find(tkey);  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  cmm.find(tkey); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  mm.count(key);  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  mm.count(tkey); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  mm.contains(key);   // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  cmm.contains(key);  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  mm.contains(tkey);  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  cmm.contains(tkey); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  mm.lower_bound(key);   // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  cmm.lower_bound(key);  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  mm.lower_bound(tkey);  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  cmm.lower_bound(tkey); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  mm.upper_bound(key);   // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  cmm.upper_bound(key);  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  mm.upper_bound(tkey);  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  cmm.upper_bound(tkey); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  mm.equal_range(key);   // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  cmm.equal_range(key);  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  mm.equal_range(tkey);  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  cmm.equal_range(tkey); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
}
