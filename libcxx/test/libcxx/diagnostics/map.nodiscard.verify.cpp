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
  std::map<int, int, TransparentCompare> m;
  const std::map<int, int, TransparentCompare> cm{};

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

  int key = 0;

  m[key];            // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  m[std::move(key)]; // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

#if TEST_STD_VER >= 14
  std::map<std::string, int, std::less<>> strMap;
  const std::map<std::string, int, std::less<>> cstrMap{};

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  strMap.at("zmt");
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  cstrMap.at("hkt");
#endif
  m.at(key);  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  cm.at(key); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  m.get_allocator(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  m.key_comp();      // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  m.value_comp();    // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

#if TEST_STD_VER >= 17
  m.extract(key);      // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  m.extract(m.cend()); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
#endif

  m.find(key);  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  cm.find(key); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
#if TEST_STD_VER >= 14
  TransparentKey<int> tkey;

  m.find(tkey);  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  cm.find(tkey); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
#endif

  m.count(key); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
#if TEST_STD_VER >= 14
  m.count(tkey); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
#endif

#if TEST_STD_VER >= 20
  m.contains(key);  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  m.contains(tkey); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
#endif

  m.lower_bound(key);  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  cm.lower_bound(key); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
#if TEST_STD_VER >= 14
  m.lower_bound(tkey);  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  cm.lower_bound(tkey); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
#endif

  m.upper_bound(key);  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  cm.upper_bound(key); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
#if TEST_STD_VER >= 14
  m.upper_bound(tkey);  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  cm.upper_bound(tkey); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
#endif

  m.equal_range(key);  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  cm.equal_range(key); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
#if TEST_STD_VER >= 14
  m.equal_range(tkey);  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  cm.equal_range(tkey); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
#endif
}
