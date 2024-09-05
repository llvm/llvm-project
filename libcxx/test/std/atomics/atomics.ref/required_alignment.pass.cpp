//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// static constexpr size_t required_alignment;

#include <atomic>
#include <cassert>
#include <concepts>
#include <cstddef>

template <typename T>
constexpr void check_required_alignment() {
  std::same_as<const std::size_t> decltype(auto) required_alignment = std::atomic_ref<T>::required_alignment;
  assert(required_alignment >= alignof(T));
}

constexpr bool test() {
  check_required_alignment<int>();
  check_required_alignment<float>();
  check_required_alignment<int*>();
  struct Empty {};
  check_required_alignment<Empty>();
  struct Trivial {
    int a;
  };
  check_required_alignment<Trivial>();
  return true;
}

int main(int, char**) {
  test();
  static_assert(test());
  return 0;
}
