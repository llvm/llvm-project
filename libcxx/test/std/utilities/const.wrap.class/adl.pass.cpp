//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// constant_wrapper

// [Note 1: The unnamed second template parameter to constant_wrapper is present
// to aid argument-dependent lookup ([basic.lookup.argdep]) in finding overloads
// for which constant_wrapper's wrapped value is a suitable argument, but for which
// the constant_wrapper itself is not. — end note]

#include <cassert>
#include <concepts>
#include <type_traits>
#include <utility>

#include "helpers.h"
#include "test_macros.h"

namespace MyNamespace {
struct MyType {
  int value;

  constexpr MyType(int v = 0) : value(v) {}
};

constexpr int adl_function(MyType mt) { return mt.value * 2; }

} // namespace MyNamespace

constexpr bool test() {
  {
    constexpr MyNamespace::MyType mt{21};
    std::constant_wrapper<mt> cw_mt;

    std::same_as<int> decltype(auto) result = adl_function(cw_mt);
    assert(result == 42);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
