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
  std::flat_set<int, TransparentCompare> fs;
  const std::flat_set<int, TransparentCompare> cfs;

  fs.begin();    // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  cfs.begin();   // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  fs.end();      // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  cfs.end();     // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  fs.rbegin();   // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  cfs.rbegin();  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  fs.rend();     // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  cfs.rend();    // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  cfs.cbegin();  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  cfs.cend();    // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  cfs.crbegin(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  cfs.crend();   // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  fs.empty();    // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  fs.size();     // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  fs.max_size(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  std::move(fs).extract(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  fs.key_comp();   // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  fs.value_comp(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  int key = 0;
  TransparentKey<int> tkey;

  fs.find(key);   // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  cfs.find(key);  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  fs.find(tkey);  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  cfs.find(tkey); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  fs.count(key);  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  fs.count(tkey); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  fs.contains(key);  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  fs.contains(tkey); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  fs.lower_bound(key);   // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  cfs.lower_bound(key);  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  fs.lower_bound(tkey);  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  cfs.lower_bound(tkey); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  fs.upper_bound(key);   // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  cfs.upper_bound(key);  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  fs.upper_bound(tkey);  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  cfs.upper_bound(tkey); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  fs.equal_range(key);   // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  cfs.equal_range(key);  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  fs.equal_range(tkey);  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  cfs.equal_range(tkey); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
}
