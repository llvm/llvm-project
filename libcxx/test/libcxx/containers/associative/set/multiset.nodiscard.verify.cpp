//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Check that functions are marked [[nodiscard]]

#include <set>

#include "test_macros.h"

#if TEST_STD_VER >= 14
struct TransparentKey {
  explicit operator int() const;
};

struct TransparentCompare {
  using is_transparent = void; // This makes the comparator transparent

  bool operator()(const int&, const TransparentKey&) const;

  bool operator()(const TransparentKey&, const int&) const;

  bool operator()(const int&, const int&) const;
};
#endif

void test() {
  std::multiset<int> s;
  const std::multiset<int> cs;

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
  std::multiset<int, TransparentCompare> ts;
  const std::multiset<int, TransparentCompare> cts{};

  TransparentKey tkey;

  ts.find(tkey);  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  cts.find(tkey); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
#endif

  s.count(key); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
#if TEST_STD_VER >= 14
  ts.count(tkey); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
#endif

#if TEST_STD_VER >= 20
  s.contains(key);   // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  ts.contains(tkey); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
#endif

  s.lower_bound(key);  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  cs.lower_bound(key); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
#if TEST_STD_VER >= 14
  ts.lower_bound(tkey);  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  cts.lower_bound(tkey); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
#endif

  s.upper_bound(key);  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  cs.upper_bound(key); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
#if TEST_STD_VER >= 14
  ts.upper_bound(tkey);  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  cts.upper_bound(tkey); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
#endif

  s.equal_range(key);  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  cs.equal_range(key); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
#if TEST_STD_VER >= 14
  ts.equal_range(tkey);  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  cts.equal_range(tkey); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
#endif
}
