//===----------------------------------------------------------------------===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20
// ADDITIONAL_COMPILE_FLAGS: -Xclang -verify-ignore-unexpected=error

// constexpr void value() &&;
// Mandates: is_copy_constructible_v<E> is true and is_move_constructible_v<E> is true.

#include <expected>

#include "MoveOnly.h"

void test() {
  // MoveOnly type as error_type
  std::expected<void, MoveOnly> e(std::unexpect, 5);

  std::move(e)
      .value(); // expected-note{{in instantiation of member function 'std::expected<void, MoveOnly>::value' requested here}}
  // expected-error-re@*:* {{static assertion failed due to requirement 'is_copy_constructible_v<MoveOnly>': error_type has to be both copy constructible and move constructible}}
}

int main(int, char**) { test(); }
