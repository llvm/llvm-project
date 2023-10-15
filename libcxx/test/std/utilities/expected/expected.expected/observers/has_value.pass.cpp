//===----------------------------------------------------------------------===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// constexpr bool has_value() const noexcept;

#include <cassert>
#include <concepts>
#include <expected>
#include <optional>
#include <type_traits>
#include <utility>

#include "test_macros.h"

// Test noexcept
template <class T>
concept HasValueNoexcept =
    requires(T t) {
      { t.has_value() } noexcept;
    };

struct Foo {};
static_assert(!HasValueNoexcept<Foo>);

static_assert(HasValueNoexcept<std::expected<int, int>>);
static_assert(HasValueNoexcept<const std::expected<int, int>>);

// This type has one byte of tail padding where `std::expected` will put its
// "has value" flag. The constructor will clobber all bytes including the
// tail padding. With this type we can check that `std::expected` will set
// its "has value" flag _after_ the value/error object is constructed.
template <int c>
struct tail_clobberer {
  constexpr tail_clobberer() {
    if (!std::is_constant_evaluated()) {
      // This `memset` might actually be UB (?) but suffices to reproduce bugs
      // related to the "has value" flag.
      std::memset(this, c, sizeof(*this));
    }
  }
  alignas(2) bool b;
};

constexpr bool test() {
  // has_value
  {
    const std::expected<int, int> e(5);
    assert(e.has_value());
  }

  // !has_value
  {
    const std::expected<int, int> e(std::unexpect, 5);
    assert(!e.has_value());
  }

  // The following tests check that the "has_value" flag is not overwritten
  // by the constructor of the value. This could happen because the flag is
  // stored in the tail padding of the value.
  //
  // The first test is a simplified version of the real code where this was
  // first observed.
  //
  // The other tests use a synthetic struct that clobbers its tail padding
  // on construction, making the issue easier to reproduce.
  //
  // See https://github.com/llvm/llvm-project/issues/68552 and the linked PR.
  {
    static constexpr auto f1 = [] -> std::expected<std::optional<int>, long> { return 0; };

    static constexpr auto f2 = [] -> std::expected<std::optional<int>, int> {
      return f1().transform_error([](auto) { return 0; });
    };

    auto e = f2();
    assert(e.has_value());
  }
  {
    const std::expected<tail_clobberer<0>, bool> e = {};
    static_assert(sizeof(tail_clobberer<0>) == sizeof(e));
    assert(e.has_value());
  }
  {
    const std::expected<void, tail_clobberer<1>> e(std::unexpect);
    static_assert(sizeof(tail_clobberer<1>) == sizeof(e));
    assert(!e.has_value());
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());
  return 0;
}
