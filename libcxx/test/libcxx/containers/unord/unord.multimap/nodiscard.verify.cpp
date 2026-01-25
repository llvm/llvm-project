//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Check that functions are marked [[nodiscard]]

#include <unordered_map>
#include <utility>

#include "test_macros.h"

struct TransparentKey {};

struct StoredKey {
  friend bool operator==(StoredKey const&, StoredKey const&) { return true; }
  friend bool operator==(StoredKey const&, TransparentKey const&) { return true; }
};

struct TransparentKeyHash {
  using is_transparent = void;

  std::size_t operator()(TransparentKey const&) const { return 0; }
  std::size_t operator()(StoredKey const&) const { return 0; }
};

void test() {
  std::unordered_multimap<int, int> m;
  const std::unordered_multimap<int, int> cm;

  m.get_allocator(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  m.empty();    // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  m.size();     // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  m.max_size(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  m.begin();   // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  m.end();     // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  cm.begin();  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  cm.end();    // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  cm.cbegin(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  cm.cend();   // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  int key = 0;

#if TEST_STD_VER >= 17
  m.extract(0);         // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  m.extract(m.begin()); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
#endif

  m.hash_function(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  m.key_eq();        // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  m.find(key);  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  cm.find(key); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
#if TEST_STD_VER >= 20
  std::unordered_multimap<StoredKey, int, TransparentKeyHash, std::equal_to<>> tm;
  const std::unordered_multimap<StoredKey, int, TransparentKeyHash, std::equal_to<>> ctm;

  TransparentKey tkey;

  tm.find(tkey);  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  ctm.find(tkey); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
#endif

#if TEST_STD_VER >= 20
  m.count(key);  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  cm.count(key); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  tm.contains(tkey);  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  ctm.contains(tkey); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
#endif

  m.equal_range(key);  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  cm.equal_range(key); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
#if TEST_STD_VER >= 20
  tm.equal_range(tkey);  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  ctm.equal_range(tkey); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
#endif

  m.bucket_count();     // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  m.max_bucket_count(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  int size = 0;

  m.bucket_size(size); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  m.bucket(key);       // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  m.begin(size);   // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  m.end(size);     // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  cm.begin(size);  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  cm.end(size);    // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  cm.cbegin(size); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  cm.cend(size);   // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  m.load_factor();     // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  m.max_load_factor(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
}
