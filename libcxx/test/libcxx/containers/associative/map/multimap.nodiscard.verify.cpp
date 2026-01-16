//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Check that functions are marked [[nodiscard]]

#include <map>
#include <string>
#include <utility>

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
#endif // TEST_STD_VER >= 14

void test() {
  std::multimap<int, int> m;
  const std::multimap<int, int> cm;

  m.begin();  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  cm.begin(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  m.end();    // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  cm.end();   // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  m.rbegin();  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  cm.rbegin(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  m.rend();    // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  cm.rend();   // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  cm.cbegin();  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  cm.cend();    // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  cm.crbegin(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  cm.crend();   // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  m.empty();    // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  m.size();     // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  m.max_size(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  m.get_allocator(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  m.key_comp();      // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  m.value_comp();    // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  int key = 0;

#if TEST_STD_VER >= 17
  m.extract(key);      // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  m.extract(m.cend()); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
#endif

  m.find(key);  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  cm.find(key); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
#if TEST_STD_VER >= 14
  std::multimap<int, int, TransparentCompare> tm;
  const std::multimap<int, int, TransparentCompare> ctm{};

  TransparentKey tkey;

  tm.find(tkey);  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  ctm.find(tkey); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
#endif

  m.count(key); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
#if TEST_STD_VER >= 14
  tm.count(tkey); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
#endif

#if TEST_STD_VER >= 20
  m.contains(key);   // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  tm.contains(tkey); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
#endif

  m.lower_bound(key);  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  cm.lower_bound(key); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
#if TEST_STD_VER >= 14
  tm.lower_bound(tkey);  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  ctm.lower_bound(tkey); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
#endif

  m.upper_bound(key);  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  cm.upper_bound(key); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
#if TEST_STD_VER >= 14
  tm.upper_bound(tkey);  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  ctm.upper_bound(tkey); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
#endif

  m.equal_range(key);  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  cm.equal_range(key); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
#if TEST_STD_VER >= 14
  tm.equal_range(tkey);  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  ctm.equal_range(tkey); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
#endif
}
