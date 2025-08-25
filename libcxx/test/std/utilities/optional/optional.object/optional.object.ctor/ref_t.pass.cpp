//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// <optional>

#include <cassert>
#include <optional>
#include <type_traits>
#include <utility>

template <typename RefType, std::remove_reference_t<RefType> _Val>
constexpr bool test() {
  std::remove_reference_t<RefType> item{_Val};
  std::optional<RefType> opt{item};

  {
    assert(*opt == item);
    assert(&(*opt) == &item);
  }
  {
    assert(*std::as_const(opt) == item);
    assert(&(*std::as_const(opt)) == &item);
  }

  return true;
}

int main(int, char**) {
  static_assert((test<int&, 1>()));
  static_assert((test<double&, 1.0>()));
  assert((test<int&, 1>()));
  assert((test<double&, 1.0>()));
}
