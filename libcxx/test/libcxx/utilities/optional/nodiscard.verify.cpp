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

struct LVal {
  constexpr std::optional<int> operator()(int&) { return 1; }
  std::optional<int> operator()(const int&)  = delete;
  std::optional<int> operator()(int&&)       = delete;
  std::optional<int> operator()(const int&&) = delete;
};

struct CLVal {
  std::optional<int> operator()(int&) = delete;
  constexpr std::optional<int> operator()(const int&) { return 1; }
  std::optional<int> operator()(int&&)       = delete;
  std::optional<int> operator()(const int&&) = delete;
};

struct RVal {
  std::optional<int> operator()(int&)       = delete;
  std::optional<int> operator()(const int&) = delete;
  constexpr std::optional<int> operator()(int&&) { return 1; }
  std::optional<int> operator()(const int&&) = delete;
};

struct CRVal {
  std::optional<int> operator()(int&)       = delete;
  std::optional<int> operator()(const int&) = delete;
  std::optional<int> operator()(int&&)      = delete;
  constexpr std::optional<int> operator()(const int&&) { return 1; }
};

void test() {
  std::bad_optional_access ex;

  ex.what(); // expect-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  std::optional<int> opt;
  const std::optional<int> cOpt;

  opt.has_value(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

#if TEST_STD_VER >= 26
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
  opt.value_or(82);
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  cOpt.value_or(82);
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::move(opt).value_or(82);
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::move(cOpt).value_or(82);

#if TEST_STD_VER >= 23
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  opt.and_then(LVal{});
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  cOpt.and_then(CLVal{});
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::move(opt).and_then(RVal{});
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::move(cOpt).and_then(CRVal{});

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  opt.transform(LVal{});
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  cOpt.transform(CLVal{});
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::move(opt).transform(RVal{});
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::move(cOpt).transform(CRVal{});

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  opt.or_else([] { return std::optional<int>{82}; });
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::move(opt).or_else([] { return std::optional<int>{82}; });
#endif // TEST_STD_VER >= 23

#if TEST_STD_VER >= 26
  int z = 94;
  std::optional<int&> optRef{z};
  const std::optional<int&> cOptRef{z};

  optRef.has_value(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  optRef.begin(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  optRef.end();   // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  *optRef;  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  *cOptRef; // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  optRef.value(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::move(optRef).value_or(z);
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  cOptRef.value_or(z);

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  optRef.and_then([](int&) { return std::optional<int>{82}; });
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  cOptRef.and_then([](int&) { return std::optional<int>{82}; });

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  optRef.transform([](int&) { return std::optional<int>{82}; });
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  cOptRef.transform([](int&) { return std::optional<int>{82}; });

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  optRef.or_else([] { return std::optional<int&>{}; });
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::move(optRef).or_else([] { return std::optional<int&>{}; });
#endif // TEST_STD_VER >= 26

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::make_optional(82);
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::make_optional<int>('h');
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::make_optional<std::string>({'z', 'm', 't'});

  std::hash<std::optional<int>> hash;

  hash(opt); //expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
}
