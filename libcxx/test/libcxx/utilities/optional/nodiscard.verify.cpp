//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++17

// <optional>

// Check that functions are marked [[nodiscard]]

#include <string>
#include <optional>
#include <utility>

#include "test_macros.h"

void test() {
  // [optional.bad.access]

  std::bad_optional_access ex;

  ex.what(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  // [optional.optional]

  std::optional<int> opt;
  const std::optional<int> cOpt;

  opt.has_value(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

#if TEST_STD_VER >= 26 && _LIBCPP_HAS_EXPERIMENTAL_OPTIONAL_ITERATOR
  opt.begin();  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  cOpt.begin(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  opt.end();    // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  cOpt.end();   // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
#endif

  *opt;             // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  *cOpt;            // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  *std::move(opt);  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  *std::move(cOpt); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  opt.value();             // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  cOpt.value();            // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::move(opt).value();  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::move(cOpt).value(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  opt.value_or(94);
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  cOpt.value_or(94);
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::move(opt).value_or(94);
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::move(cOpt).value_or(94);

#if TEST_STD_VER >= 23
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  opt.and_then([](int&) { return std::optional<int>{82}; });
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  cOpt.and_then([](const int&) { return std::optional<int>{82}; });
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::move(opt).and_then([](int&&) { return std::optional<int>{82}; });
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::move(cOpt).and_then([](const int&&) { return std::optional<int>{82}; });

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  opt.transform([](int&) { return std::optional<int>{94}; });
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  cOpt.transform([](const int&) { return std::optional<int>{94}; });
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::move(opt).transform([](int&&) { return std::optional<int>{94}; });
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::move(cOpt).transform([](const int&&) { return std::optional<int>{94}; });

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  opt.or_else([] { return std::optional<int>{82}; });
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::move(opt).or_else([] { return std::optional<int>{82}; });
#endif // TEST_STD_VER >= 23

  // [optional.optional.ref]

#if TEST_STD_VER >= 26
  int z = 94;
  std::optional<int&> optRef{z};
  const std::optional<int&> cOptRef{z};

  optRef.has_value(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

#  if _LIBCPP_HAS_EXPERIMENTAL_OPTIONAL_ITERATOR
  optRef.begin(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  optRef.end();   // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
#  endif

  *optRef;  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  *cOptRef; // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  optRef.value(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::move(optRef).value_or(z);
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  cOptRef.value_or(z);

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  optRef.and_then([](int&) { return std::optional<int>{94}; });
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  cOptRef.and_then([](int&) { return std::optional<int>{94}; });

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  optRef.transform([](int&) { return std::optional<int>{82}; });
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  cOptRef.transform([](int&) { return std::optional<int>{82}; });

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  optRef.or_else([] { return std::optional<int&>{}; });
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::move(optRef).or_else([] { return std::optional<int&>{}; });
#endif // TEST_STD_VER >= 26

  // [optional.specalg]

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::make_optional(82);
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::make_optional<int>('h');
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::make_optional<std::string>({'z', 'm', 't'});

  // [optional.hash]

  std::hash<std::optional<int>> hash;

  hash(opt); //expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
}
