//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// constexpr W operator*() const noexcept(is_nothrow_copy_constructible_v<W>);

#include "test_macros.h"

TEST_CLANG_DIAGNOSTIC_IGNORED("-Wsign-compare")
TEST_GCC_DIAGNOSTIC_IGNORED("-Wsign-compare")
TEST_MSVC_DIAGNOSTIC_IGNORED(4018) // various "signed/unsigned mismatch"

#include <ranges>
#include <cassert>

#include "../types.h"

struct NotNoexceptCopy {
  using difference_type = int;

  int value_;
  constexpr explicit NotNoexceptCopy(int value = 0) : value_(value) {}
  NotNoexceptCopy(const NotNoexceptCopy&) noexcept(false) = default;

  bool operator==(const NotNoexceptCopy&) const = default;

  friend constexpr NotNoexceptCopy& operator+=(NotNoexceptCopy &lhs, const NotNoexceptCopy& rhs) {
    lhs.value_ += rhs.value_; return lhs;
  }
  friend constexpr NotNoexceptCopy& operator-=(NotNoexceptCopy &lhs, const NotNoexceptCopy& rhs) {
    lhs.value_ -= rhs.value_; return lhs;
  }

  friend constexpr NotNoexceptCopy operator+(NotNoexceptCopy lhs, NotNoexceptCopy rhs) {
    return NotNoexceptCopy{lhs.value_ + rhs.value_};
  }
  friend constexpr int operator-(NotNoexceptCopy lhs, NotNoexceptCopy rhs) {
    return lhs.value_ - rhs.value_;
  }

  constexpr NotNoexceptCopy& operator++()     { ++value_; return *this; }
  constexpr void              operator++(int) { ++value_;               }
};

template<class T>
constexpr void testType() {
  {
    std::ranges::iota_view<T> io(T(0));
    auto iter = io.begin();
    for (int i = 0; i < 100; ++i, ++iter)
      assert(*iter == T(i));

    static_assert(noexcept(*iter) == !std::same_as<T, NotNoexceptCopy>);
  }
  {
    std::ranges::iota_view<T> io(T(10));
    auto iter = io.begin();
    for (int i = 10; i < 100; ++i, ++iter)
      assert(*iter == T(i));
  }
  {
    const std::ranges::iota_view<T> io(T(0));
    auto iter = io.begin();
    for (int i = 0; i < 100; ++i, ++iter)
      assert(*iter == T(i));
  }
  {
    const std::ranges::iota_view<T> io(T(10));
    auto iter = io.begin();
    for (int i = 10; i < 100; ++i, ++iter)
      assert(*iter == T(i));
  }
}

constexpr bool test() {
  testType<SomeInt>();
  testType<NotNoexceptCopy>();
  testType<signed long>();
  testType<unsigned long>();
  testType<int>();
  testType<unsigned>();
  testType<short>();
  testType<unsigned short>();

  // Tests a mix of signed unsigned types.
  {
    const std::ranges::iota_view<int, unsigned> io(0, 10);
    auto iter = io.begin();
    for (int i = 0; i < 10; ++i, ++iter)
      assert(*iter == i);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
