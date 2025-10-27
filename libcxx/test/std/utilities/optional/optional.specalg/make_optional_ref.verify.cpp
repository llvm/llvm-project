//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// <optional>

// template <class T, class... Args>
//   constexpr optional<T> make_optional(Args&&... args);

#include <optional>

struct Foo {
  int x, y;
};

struct X {
  double i_;

public:
  explicit X(int& i) : i_(i) {}
};

int main(int, char**) {
  int i = 1;
  // expected-error-re@optional:* 4 {{static assertion failed{{.*}} make_optional<T&, Args...> is disallowed}}
  std::make_optional<int&>(i);
  std::make_optional<X&>(i);
  std::make_optional<int&>(1);
  std::make_optional<Foo&>(1, 2);

  // FIXME: Garbage error messages that Clang produces after the static_assert is reported
  // expected-error-re@optional:* 0+ {{no matching constructor for initialization of 'optional<{{.*}}>'}}
}