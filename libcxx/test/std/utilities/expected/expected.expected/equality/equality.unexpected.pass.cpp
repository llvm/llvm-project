//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// template<class E2> friend constexpr bool operator==(const expected& x, const unexpected<E2>& e);

#include <cassert>
#include <concepts>
#include <expected>
#include <type_traits>
#include <utility>

#include "test_comparisons.h"
#include "test_macros.h"

#if TEST_STD_VER >= 26
// https://wg21.link/P3379R0
static_assert(HasOperatorEqual<std::expected<EqualityComparable, EqualityComparable>, std::unexpected<int>>);
static_assert(HasOperatorEqual<std::expected<EqualityComparable, int>, std::unexpected<EqualityComparable>>);
static_assert(!HasOperatorEqual<std::expected<EqualityComparable, NonComparable>, std::unexpected<int>>);
#endif

constexpr bool test() {
  // x.has_value()
  {
    const std::expected<EqualityComparable, EqualityComparable> e1(std::in_place, 5);
    std::unexpected<int> un2(10);
    std::unexpected<int> un3(5);
    assert(e1 != un2);
    assert(e1 != un3);
  }

  // !x.has_value()
  {
    const std::expected<EqualityComparable, EqualityComparable> e1(std::unexpect, 5);
    std::unexpected<int> un2(10);
    std::unexpected<int> un3(5);
    assert(e1 != un2);
    assert(e1 == un3);
  }

  // LWG4366
  { // x.has_value()
    const std::expected<int, ImplicitBool::E1> e1(std::in_place, 1);
    const std::unexpected<ImplicitBool::E2> u2{ImplicitBool::E2{1}};
    const std::unexpected<ImplicitBool::E2> u3{ImplicitBool::E2{2}};

    assert(e1 != u2);
    assert(e1 != u3);
  }

  { // !x.has_value()
    const std::unexpected<ImplicitBool::E1> u1{ImplicitBool::E1{1}};
    const std::unexpected<ImplicitBool::E2> u2{ImplicitBool::E2{1}};
    const std::unexpected<ImplicitBool::E2> u3{ImplicitBool::E2{2}};

    const std::expected<int, ImplicitBool::E1> e1(u1);
    assert(e1 == u2);
    assert(e1 != u3);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());
  return 0;
}
