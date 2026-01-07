//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Check that functions are marked [[nodiscard]]

#include <unordered_set>
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
  std::unordered_multiset<int> us;
  const std::unordered_multiset<int> cus;

  us.get_allocator(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  us.empty();    // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  us.size();     // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  us.max_size(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  us.begin();   // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  us.end();     // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  cus.begin();  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  cus.end();    // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  cus.cbegin(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  cus.cend();   // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  int key = 0;

#if TEST_STD_VER >= 17
  us.extract(key);        // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  us.extract(us.begin()); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
#endif

  us.hash_function(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  us.key_eq();        // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  us.find(key);  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  cus.find(key); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
#if TEST_STD_VER >= 20
  std::unordered_multiset<StoredKey, TransparentKeyHash, std::equal_to<>> tus;
  const std::unordered_multiset<StoredKey, TransparentKeyHash, std::equal_to<>> ctus;

  TransparentKey tkey;

  tus.find(tkey);  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  ctus.find(tkey); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
#endif

  us.count(key); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
#if TEST_STD_VER >= 20
  tus.count(tkey); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
#endif

#if TEST_STD_VER >= 20
  us.contains(key);   // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  tus.contains(tkey); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
#endif

  us.equal_range(key);  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  cus.equal_range(key); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
#if TEST_STD_VER >= 20
  tus.equal_range(tkey);  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  ctus.equal_range(tkey); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
#endif

  us.bucket_count();     // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  us.max_bucket_count(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  int size = 0;

  us.bucket_size(size); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  us.bucket(key);       // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  us.begin(size);   // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  us.end(size);     // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  cus.begin(size);  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  cus.end(size);    // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  cus.cbegin(size); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  cus.cend(size);   // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  us.load_factor();     // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  us.max_load_factor(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
}
