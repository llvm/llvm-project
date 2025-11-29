//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// Check that functions are marked [[nodiscard]]

#include <set>

#include "test_macros.h"

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
  std::set<int, TransparentCompare> s;
  const std::set<int, TransparentCompare> cs{};

  s.begin();  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  cs.begin(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  s.end();    // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  cs.end();   // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  s.rbegin();  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  cs.rbegin(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  s.rend();    // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  cs.rend();   // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  cs.cbegin();  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  cs.cend();    // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  cs.crbegin(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  cs.crend();   // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  s.empty();    // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  s.size();     // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  s.max_size(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  int key = 0;

#if TEST_STD_VER >= 17
  s.extract(key);       // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  s.extract(s.begin()); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
#endif

  s.get_allocator(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  s.key_comp();      // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  s.value_comp();    // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  s.find(key);  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  cs.find(key); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
#if TEST_STD_VER >= 14
  TransparentKey<int> tkey;

  s.find(tkey);  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  cs.find(tkey); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
#endif

  s.count(key); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
#if TEST_STD_VER >= 14
  s.count(tkey); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
#endif

  s.lower_bound(key);  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  cs.lower_bound(key); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
#if TEST_STD_VER >= 14
  s.lower_bound(tkey);  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  cs.lower_bound(tkey); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
#endif

  s.upper_bound(key);  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  cs.upper_bound(key); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
#if TEST_STD_VER >= 14
  s.upper_bound(tkey);  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  cs.upper_bound(tkey); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
#endif

  s.equal_range(key);  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  cs.equal_range(key); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
#if TEST_STD_VER >= 14
  s.equal_range(tkey);  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  cs.equal_range(tkey); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
#endif
}
