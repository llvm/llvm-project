//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++20

// <numeric>

// template <class _Tp>
// _Tp midpoint(_Tp __a, _Tp __b) noexcept
// Constraints:
//   - T is an arithmetic type other than bool.

#include <cassert>
#include <concepts>
#include <cstddef>
#include <limits>
#include <numeric>

#include "test_macros.h"
#include "type_algorithms.h"

template <typename T>
concept has_midpoint = requires(T a, T b) { std::midpoint(a, b); };

static_assert(!has_midpoint<bool>);
static_assert(!has_midpoint<const bool>);
static_assert(!has_midpoint<volatile bool>);
static_assert(!has_midpoint<const volatile bool>);

template <typename T>
constexpr void test_signed() {
  ASSERT_SAME_TYPE(decltype(std::midpoint(T{}, T{})), T);
  ASSERT_NOEXCEPT(std::midpoint(T{}, T{}));
  using limits = std::numeric_limits<T>;

  assert(std::midpoint(T{1}, T{3}) == T{2});
  assert(std::midpoint(T{3}, T{1}) == T{2});

  assert(std::midpoint(T{0}, T{0}) == T{0});
  assert(std::midpoint(T{0}, T{2}) == T{1});
  assert(std::midpoint(T{2}, T{0}) == T{1});
  assert(std::midpoint(T{2}, T{2}) == T{2});

  assert(std::midpoint(T{1}, T{4}) == T{2});
  assert(std::midpoint(T{4}, T{1}) == T{3});
  assert(std::midpoint(T{3}, T{4}) == T{3});
  assert(std::midpoint(T{4}, T{3}) == T{4});

  assert(std::midpoint(T{-3}, T{4}) == T{0});
  assert(std::midpoint(T{-4}, T{3}) == T{-1});
  assert(std::midpoint(T{3}, T{-4}) == T{0});
  assert(std::midpoint(T{4}, T{-3}) == T{1});
  assert(std::midpoint(T{-3}, T{-4}) == T{-3});
  assert(std::midpoint(T{-4}, T{-3}) == T{-4});

  assert(std::midpoint(limits::min(), limits::max()) == T{-1});
  assert(std::midpoint(limits::max(), limits::min()) == T{0});

  assert(std::midpoint(limits::min(), T{6}) == T{limits::min() / 2 + 3});
  assert(std::midpoint(T{6}, limits::min()) == T{limits::min() / 2 + 3});

  assert(std::midpoint(limits::max(), T{6}) == T{limits::max() / 2 + 4});
  assert(std::midpoint(T{6}, limits::max()) == T{limits::max() / 2 + 3});

  assert(std::midpoint(limits::min(), T{-6}) == T{limits::min() / 2 - 3});
  assert(std::midpoint(T{-6}, limits::min()) == T{limits::min() / 2 - 3});

  assert(std::midpoint(limits::max(), T{-6}) == T{limits::max() / 2 - 2});
  assert(std::midpoint(T{-6}, limits::max()) == T{limits::max() / 2 - 3});
}

template <typename T>
constexpr void test_unsigned() {
  ASSERT_SAME_TYPE(decltype(std::midpoint(T{}, T{})), T);
  ASSERT_NOEXCEPT(std::midpoint(T{}, T{}));

  using limits     = std::numeric_limits<T>;
  const T half_way = (limits::max() - limits::min()) / 2;

  assert(std::midpoint(T{1}, T{3}) == T{2});
  assert(std::midpoint(T{3}, T{1}) == T{2});

  assert(std::midpoint(T{0}, T{0}) == T{0});
  assert(std::midpoint(T{0}, T{2}) == T{1});
  assert(std::midpoint(T{2}, T{0}) == T{1});
  assert(std::midpoint(T{2}, T{2}) == T{2});

  assert(std::midpoint(T{1}, T{4}) == T{2});
  assert(std::midpoint(T{4}, T{1}) == T{3});
  assert(std::midpoint(T{3}, T{4}) == T{3});
  assert(std::midpoint(T{4}, T{3}) == T{4});

  assert(std::midpoint(limits::min(), limits::max()) == half_way);
  assert(std::midpoint(limits::max(), limits::min()) == T{half_way + 1});

  assert(std::midpoint(limits::min(), T{6}) == T{3});
  assert(std::midpoint(T{6}, limits::min()) == T{3});

  assert(std::midpoint(limits::max(), T{6}) == T{half_way + 4});
  assert(std::midpoint(T{6}, limits::max()) == T{half_way + 3});
}

constexpr bool test() {
  types::for_each(types::integer_types(), []<class T>() {
    if constexpr (!std::same_as<T, bool>) {
      if constexpr (std::signed_integral<T>) {
        test_signed<T>();
      } else {
        test_unsigned<T>();
      }
    }
  });
  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
