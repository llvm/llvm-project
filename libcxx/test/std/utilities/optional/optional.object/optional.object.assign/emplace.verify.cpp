//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26
// <optional>

// template <class U> T& optional<T&>::emplace(U&& arg);
// Ensure that emplace isn't found if reference_constructs_From_temporary_v == true or is_constructible_v<_Tp&, U> == false
#include <optional>
#include <type_traits>

struct X {};
void test() {
  int i = 1;
  std::optional<X&> f{};
  static_assert(!std::is_constructible_v<X&, int>); // is_constructible_v<_Tp&, U> == false
  f.emplace(i); // expected-error {{no matching member function for call to 'emplace'}}

  std::optional<const int&> t{}; // reference_constructs_from_temporary_v == false
  t.emplace(1);                  // expected-error {{no matching member function for call to 'emplace'}}
}
