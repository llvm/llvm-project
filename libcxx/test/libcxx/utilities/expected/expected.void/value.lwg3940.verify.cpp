//===----------------------------------------------------------------------===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20
// UNSUPPORTED: no-rtti
// UNSUPPORTED: no-exceptions

// constexpr void value() const &;
// Mandates: is_copy_constructible_v<E> is true.

// constexpr void value() &&;
// Mandates: is_copy_constructible_v<E> is true and is_move_constructible_v<E> is true.

#include <expected>

#include "MoveOnly.h"

struct CopyOnly {
  CopyOnly()                = default;
  CopyOnly(const CopyOnly&) = default;
  CopyOnly(CopyOnly&&)      = delete;
};

void test() {
  // MoveOnly type as error_type
  std::expected<void, MoveOnly> e(std::unexpect, 5);

  e.value(); // expected-note {{in instantiation of member function 'std::expected<void, MoveOnly>::value' requested here}}
  // expected-error@*:* {{static assertion failed due to requirement 'is_copy_constructible_v<MoveOnly>'}}
  // expected-error@*:* {{call to deleted constructor of 'MoveOnly'}}

  std::move(e)
      .value(); // expected-note {{in instantiation of member function 'std::expected<void, MoveOnly>::value' requested here}}
  // expected-error@*:* {{static assertion failed due to requirement 'is_copy_constructible_v<MoveOnly>'}}

  // CopyOnly type as error_type
  std::expected<void, CopyOnly> e2(std::unexpect);
  // expected-error@*:* {{call to deleted constructor of 'CopyOnly'}}

  e2.value();

  std::move(e2)
      .value(); // expected-note {{in instantiation of member function 'std::expected<void, CopyOnly>::value' requested here}}
  // expected-error@*:* {{static assertion failed due to requirement 'is_move_constructible_v<CopyOnly>'}}
  // expected-error@*:* {{call to deleted constructor of 'CopyOnly'}}
}
