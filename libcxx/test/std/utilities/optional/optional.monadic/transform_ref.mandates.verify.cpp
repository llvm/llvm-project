//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// <optional>

// Test the Mandates added by LWG4299.
// template<class F> constexpr optional<remove_cv_t<invoke_result_t<F, T&>>> optional<T&>::transform(F&& f) const;
// Mandates:
//  Let U be remove_cv_t<invoke_result_t<F, T&>>.
//  U is a valid contained type for optional.

#include <optional>
#include <utility>

void test() {
  int i = 0;
  std::optional<int&> opt(i);

  // U is an rvalue reference type.
  {
    (void)opt.transform([](int& x) -> int&& { return std::move(x); });
    // expected-error-re@*:* {{static assertion failed{{.*}}Result of f(value()) should be a valid contained type for optional}}
    // expected-error-re@*:* 0+ {{static assertion failed{{.*}}instantiation of optional with an rvalue reference type is ill-formed}}
    // expected-error-re@*:* 0+ {{{{.*}}is not a direct or virtual base of{{.*}}}}
  }

  // U is void.
  {
    (void)opt.transform([](int&) {});
    // expected-error-re@*:* {{static assertion failed{{.*}}Result of f(value()) should be a valid contained type for optional}}
    // expected-error-re@*:* 0+ {{static assertion failed{{.*}}instantiation of optional with a non-object type is undefined behavior}}
    // expected-error-re@*:* 0+ {{static assertion failed{{.*}}instantiation of optional with a non-destructible type is ill-formed}}
    // expected-error@*:* 0+ {{cannot form a reference to 'void'}}
  }

  // U is in_place_t.
  {
    (void)opt.transform([](int&) { return std::in_place; });
    // expected-error-re@*:* {{static assertion failed{{.*}}Result of f(value()) should not be std::in_place_t}}
    // expected-error-re@*:* 0+ {{static assertion failed{{.*}}instantiation of optional with in_place_t is ill-formed}}
    // expected-error-re@*:* 0+ {{no matching constructor for initialization of{{.*}}}}
  }

  // U is nullopt_t.
  {
    (void)opt.transform([](int&) { return std::nullopt; });
    // expected-error-re@*:* {{static assertion failed{{.*}}Result of f(value()) should not be std::nullopt_t}}
    // expected-error-re@*:* 0+ {{static assertion failed{{.*}}instantiation of optional with nullopt_t is ill-formed}}
  }
}
