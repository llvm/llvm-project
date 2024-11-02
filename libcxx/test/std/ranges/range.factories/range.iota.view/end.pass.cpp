//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

#include "test_macros.h"

TEST_CLANG_DIAGNOSTIC_IGNORED("-Wsign-compare")
TEST_GCC_DIAGNOSTIC_IGNORED("-Wsign-compare")
TEST_MSVC_DIAGNOSTIC_IGNORED(4018 4389) // various "signed/unsigned mismatch"

// constexpr auto end() const;
// constexpr iterator end() const requires same_as<W, Bound>;

#include <cassert>
#include <ranges>
#include <utility>

#include "types.h"

template<class T, class U>
constexpr void testType(U u) {
  {
    std::ranges::iota_view<T, U> io(T(0), u);
    assert(std::ranges::next(io.begin(), 10) == io.end());
  }
  {
    std::ranges::iota_view<T, U> io(T(10), u);
    assert(io.begin() == io.end());
    assert(io.begin() == std::move(io).end());
  }
  {
    const std::ranges::iota_view<T, U> io(T(0), u);
    assert(std::ranges::next(io.begin(), 10) == io.end());
    assert(std::ranges::next(io.begin(), 10) == std::move(io).end());
  }
  {
    const std::ranges::iota_view<T, U> io(T(10), u);
    assert(io.begin() == io.end());
  }

  {
    std::ranges::iota_view<T> io(T(0), std::unreachable_sentinel);
    assert(io.begin() != io.end());
    assert(std::ranges::next(io.begin()) != io.end());
    assert(std::ranges::next(io.begin(), 10) != io.end());
  }
  {
    const std::ranges::iota_view<T> io(T(0), std::unreachable_sentinel);
    assert(io.begin() != io.end());
    assert(std::ranges::next(io.begin()) != io.end());
    assert(std::ranges::next(io.begin(), 10) != io.end());
  }
}

constexpr bool test() {
  testType<SomeInt>(SomeInt(10));
  testType<SomeInt>(IntComparableWith(SomeInt(10)));
  testType<signed long>(IntComparableWith<signed long>(10));
  testType<unsigned long>(IntComparableWith<unsigned long>(10));
  testType<int>(IntComparableWith<int>(10));
  testType<int>(int(10));
  testType<int>(unsigned(10));
  testType<unsigned>(unsigned(10));
  testType<unsigned>(int(10));
  testType<unsigned>(IntComparableWith<unsigned>(10));
  testType<short>(short(10));
  testType<short>(IntComparableWith<short>(10));
  testType<unsigned short>(IntComparableWith<unsigned short>(10));

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
