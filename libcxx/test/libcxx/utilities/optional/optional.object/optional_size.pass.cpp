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

#include <functional>
#include <locale>
#include <memory>
#include <optional>
#include <string>
#include <vector>

template <class T>
struct type_with_bool {
  T value;
  bool has_value;
};

struct alignas(void*) aligned_s {};

int main(int, char**) {
  // Test that std::optional achieves the expected size. See https://llvm.org/PR61095.
  static_assert(sizeof(std::optional<char>) == sizeof(type_with_bool<char>));
  static_assert(sizeof(std::optional<int>) == sizeof(type_with_bool<int>));
  static_assert(sizeof(std::optional<long>) == sizeof(type_with_bool<long>));
  static_assert(sizeof(std::optional<std::size_t>) == sizeof(type_with_bool<std::size_t>));

  // Check that the optionals using a tombstone have the expected size
  static_assert(sizeof(std::optional<bool>) == sizeof(bool));
  static_assert(sizeof(std::optional<std::string>) == sizeof(std::string));
  static_assert(sizeof(std::optional<int*>) == sizeof(void*));
  static_assert(sizeof(std::optional<short*>) == sizeof(void*));
  static_assert(sizeof(std::optional<char*>) == sizeof(void*) * 2);
  static_assert(sizeof(std::optional<char**>) == sizeof(void*));
  static_assert(sizeof(std::optional<aligned_s*>) == sizeof(void*) * 2);
  static_assert(sizeof(std::optional<std::unique_ptr<int>>) == sizeof(void*));
  static_assert(sizeof(std::optional<std::unique_ptr<int[]>>) == sizeof(void*));
  static_assert(sizeof(std::optional<std::unique_ptr<aligned_s>>) == sizeof(void*) * 2);
  static_assert(sizeof(std::optional<std::reference_wrapper<char*>>) == sizeof(void*));
  static_assert(sizeof(std::optional<std::reference_wrapper<aligned_s>>) == sizeof(void*));
  static_assert(sizeof(std::optional<std::locale>) == sizeof(void*));
  static_assert(sizeof(std::optional<std::shared_ptr<char>>) == sizeof(void*) * 2);
  static_assert(sizeof(std::optional<std::shared_ptr<int>>) == sizeof(void*) * 2);
  static_assert(sizeof(std::optional<std::weak_ptr<char>>) == sizeof(void*) * 2);
  static_assert(sizeof(std::optional<std::weak_ptr<int>>) == sizeof(void*) * 2);
  static_assert(sizeof(std::optional<std::pair<int, int>>) == sizeof(int) * 3);
  static_assert(sizeof(std::optional<std::pair<std::string, char*>>) == sizeof(void*) * 4);
  static_assert(sizeof(std::optional<std::pair<char*, std::string>>) == sizeof(void*) * 4);
  static_assert(sizeof(std::optional<std::vector<int>>) == sizeof(void*) * 3);
  static_assert(sizeof(std::optional<std::vector<char>>) == sizeof(void*) * 4);

  return 0;
}
