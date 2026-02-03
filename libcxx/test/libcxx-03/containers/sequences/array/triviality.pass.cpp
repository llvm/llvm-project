//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Make sure std::array<T, N> is trivially copyable whenever T is trivially copyable.
// This is not technically mandated by the Standard, but libc++ has been providing
// this property.

#include <array>
#include <type_traits>

struct Empty {};

struct TrivialCopy {
  int i;
  double j;
};

struct NonTrivialCopy {
  NonTrivialCopy(NonTrivialCopy const&) {}
  NonTrivialCopy& operator=(NonTrivialCopy const&) { return *this; }
};

template <typename T>
void check_trivially_copyable() {
  static_assert(std::is_trivially_copyable<std::array<T, 0> >::value, "");
  static_assert(std::is_trivially_copyable<std::array<T, 1> >::value, "");
  static_assert(std::is_trivially_copyable<std::array<T, 2> >::value, "");
  static_assert(std::is_trivially_copyable<std::array<T, 3> >::value, "");
}

int main(int, char**) {
  check_trivially_copyable<int>();
  check_trivially_copyable<long>();
  check_trivially_copyable<double>();
  check_trivially_copyable<long double>();
  check_trivially_copyable<Empty>();
  check_trivially_copyable<TrivialCopy>();

  // Check that std::array<T, 0> is still trivially copyable when T is not
  static_assert(std::is_trivially_copyable<std::array<NonTrivialCopy, 0> >::value, "");
  static_assert(!std::is_trivially_copyable<std::array<NonTrivialCopy, 1> >::value, "");
  static_assert(!std::is_trivially_copyable<std::array<NonTrivialCopy, 2> >::value, "");
  static_assert(!std::is_trivially_copyable<std::array<NonTrivialCopy, 3> >::value, "");

  return 0;
}
