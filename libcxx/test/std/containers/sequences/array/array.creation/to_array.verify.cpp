//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <array>
// UNSUPPORTED: c++03, c++11, c++14, c++17

#include <array>

#include "test_macros.h"
#include "MoveOnly.h"

// expected-warning@array:* 0-1 {{suggest braces around initialization of subobject}}

int main(int, char**) {
  {
    char source[3][6] = {"hi", "world"};
    // expected-error@array:* {{to_array does not accept multidimensional arrays}}
    // expected-error@array:* {{to_array requires copy constructible elements}}
    // expected-error@array:* 3 {{cannot initialize}}
    std::to_array(source); // expected-note {{requested here}}
  }

  {
    MoveOnly mo[] = {MoveOnly{3}};
    // expected-error@array:* {{to_array requires copy constructible elements}}
    // expected-error-re@array:* {{{{(call to implicitly-deleted copy constructor of 'MoveOnly')|(call to deleted constructor of 'MoveOnly')}}}}
    std::to_array(mo); // expected-note {{requested here}}
  }

  {
    const MoveOnly cmo[] = {MoveOnly{3}};
    // expected-error@array:* {{to_array requires move constructible elements}}
    // expected-error-re@array:* {{{{(call to implicitly-deleted copy constructor of 'MoveOnly')|(call to deleted constructor of 'MoveOnly')}}}}
    std::to_array(std::move(cmo)); // expected-note {{requested here}}
  }

  return 0;
}
