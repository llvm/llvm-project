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
#include "../../types.h"

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
    auto f1 = [] -> std::expected<std::optional<int>, long> { return 0; };

    auto f2 = [&f1] -> std::expected<std::optional<int>, int> {
      return f1().transform_error([](auto) { return 0; });
    };

    auto e = f2();
    assert(e.has_value());
  }
  {
    const std::expected<TailClobberer<0>, bool> e = {};
    // clang-cl does not support [[no_unique_address]] yet.
#if !(defined(TEST_COMPILER_CLANG) && defined(_MSC_VER))
    LIBCPP_STATIC_ASSERT(sizeof(TailClobberer<0>) == sizeof(e));
#endif
    assert(e.has_value());
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());
  return 0;
}
