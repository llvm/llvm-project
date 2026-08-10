//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <bitset>

// Check that functions are marked [[nodiscard]]

#include <bitset>

#include "test_macros.h"
#include "test_allocator.h"

void test() {
  std::bitset<11> bs;
  const std::bitset<11> cbs;

  // std::bitset<>::reference operator~() const noexcept;
  ~bs[0]; // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  ~bs; // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  bs[0];          // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  cbs[0];         // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  bs.to_ulong();  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  bs.to_ullong(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  struct CharTraits : public std::char_traits<char> {};

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  bs.to_string<char, CharTraits, test_allocator<char> >();
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  bs.to_string<char, CharTraits>();
#if !defined(TEST_HAS_NO_WIDE_CHARACTERS)
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  bs.to_string<wchar_t>();
#endif
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  bs.to_string();

  bs.count(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  bs.size();  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  bs.test(0); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  bs.all();   // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  bs.any();   // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  bs.none();  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  bs << 1;    // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  bs >> 1;    // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  bs & bs; // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  bs | bs; // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  bs ^ bs; // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  std::hash<std::bitset<11> > hash;

  hash(bs); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
}
