//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <atomic>

// atomic() noexcept = default;                                        // until C++20
// constexpr atomic() noexcept(is_nothrow_default_constructible_v<T>); // since C++20

// Constraints: is_default_constructible_v<T> is true. (since C++20)

#include <atomic>
#include <type_traits>

#include "test_macros.h"

template <typename T>
void test() {
#if TEST_STD_VER >= 11
  constexpr T a{};
  (void)a;
#  if TEST_STD_VER >= 20
  [[maybe_unused]] constexpr T b;
#  endif
#else
  T a;
  (void)a;
#endif
  static_assert(std::is_nothrow_constructible<T>::value, "");
  ASSERT_NOEXCEPT(T{});
}

struct throwing {
  throwing() {}
};

struct trivial {
  int a;
};

struct not_default_constructible {
  explicit not_default_constructible(int) {}
};

int main(int, char**) {
  test<std::atomic<bool> >();
  test<std::atomic<int> >();
  test<std::atomic<int*> >();
  test<std::atomic<trivial> >();
  test<std::atomic_flag>();

#if TEST_STD_VER >= 20
  static_assert(!std::is_nothrow_constructible_v<std::atomic<throwing>>);
  ASSERT_NOT_NOEXCEPT(std::atomic<throwing>{});

  static_assert(!std::is_default_constructible_v<std::atomic<not_default_constructible>>);
#else
  static_assert(std::is_nothrow_constructible<std::atomic<throwing> >::value, "");

  ASSERT_NOEXCEPT(std::atomic<throwing>{});
#  ifndef TEST_COMPILER_GCC
  static_assert(std::is_default_constructible<std::atomic<not_default_constructible> >::value, "");
#  endif
#endif

  return 0;
}
