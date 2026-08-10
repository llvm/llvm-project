//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// <optional>

// template <class T> class optional;

#include <optional>

template <class T>
struct type_with_bool {
  T value;
  bool has_value;
};

int main(int, char**) {
  // Test that std::optional achieves the expected size. See https://llvm.org/PR61095.
  static_assert(sizeof(std::optional<char>) == sizeof(type_with_bool<char>));
  static_assert(sizeof(std::optional<int>) == sizeof(type_with_bool<int>));
  static_assert(sizeof(std::optional<long>) == sizeof(type_with_bool<long>));
  static_assert(sizeof(std::optional<std::size_t>) == sizeof(type_with_bool<std::size_t>));

  return 0;
}
