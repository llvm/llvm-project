//===----------------------------------------------------------------------===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

#include <memory>

struct NonTrivial {
  int x;
  ~NonTrivial() {}
};

struct Incomplete; // expected-note {{forward declaration of 'Incomplete'}}

void test_invalid_types(void* p) {
  // expected-error@*:* {{type 'NonTrivial' is not an implicit-lifetime type, cannot start lifetime}}
  std::start_lifetime_as<NonTrivial>(p);

  // expected-error@*:* {{incomplete type 'Incomplete' where a complete type is required}}
  std::start_lifetime_as<Incomplete>(p);

  // expected-error@*:* {{type 'void' is not an implicit-lifetime type, cannot start lifetime}}
  std::start_lifetime_as<void>(p);

  // expected-error@*:* {{static_cast from 'void *' to 'void (*)()' is not allowed}}
  std::start_lifetime_as<void()>(p);
}

void test_constexpr() {
  // expected-error@+1 {{constexpr variable 'fail' must be initialized by a constant expression}}
  constexpr auto* fail = std::start_lifetime_as<int>((void*)nullptr);
}