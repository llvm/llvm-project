//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// <array>

// template <class T, size_t N>
//   constexpr array<remove_cv_t<T>, N> to_array(T (&a)[N]);
// template <class T, size_t N>
//   constexpr array<remove_cv_t<T>, N> to_array(T (&&a)[N]);

#include <array>

#include "MoveOnly.h"
#include "test_macros.h"

// expected-warning@array:* 0-1 {{suggest braces around initialization of subobject}}

void test() {
  {
    char source[3][6] = {"hi", "world"};
    // expected-error@array:* {{to_array does not accept multidimensional arrays}}
    // expected-error@array:* {{to_array requires copy constructible elements}}
    // expected-error@array:* 3 {{cannot initialize}}
    (void)std::to_array(source); // expected-note {{requested here}}
  }

  {
    MoveOnly mo[] = {MoveOnly{3}};
    // expected-error@array:* {{to_array requires copy constructible elements}}
    // expected-error-re@array:* 1-2{{{{(call to implicitly-deleted copy constructor of 'MoveOnly')|(call to deleted constructor of 'MoveOnly')}}}}
    (void)std::to_array(mo); // expected-note {{requested here}}
  }

  {
    const MoveOnly cmo[] = {MoveOnly{3}};
    // expected-error@array:* {{to_array requires move constructible elements}}
    // expected-error-re@array:* 0-1{{{{(call to implicitly-deleted copy constructor of 'MoveOnly')|(call to deleted constructor of 'MoveOnly')}}}}
    (void)std::to_array(std::move(cmo)); // expected-note {{requested here}}
  }
}
