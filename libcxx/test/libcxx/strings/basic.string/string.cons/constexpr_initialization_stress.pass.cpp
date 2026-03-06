//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++20
// REQUIRES: clang

// <string>

// Stress test for constexpr std::string initialization.
// This test ensures that we can handle a large number of constexpr strings.

// ADDITIONAL_COMPILE_FLAGS(has-fconstexpr-steps): -fconstexpr-steps=1000000
// ADDITIONAL_COMPILE_FLAGS(has-fconstexpr-ops-limit): -fconstexpr-ops-limit=1000000

#include <string>
#include <array>
#include <cassert>

consteval auto eval() {
  std::array<std::string, 6000> r;
  for (auto& s : r) {
    s = "hello";
  }
  return r;
}

void test() {
  static constexpr auto r = eval();
  assert(r.front() == "hello");
  assert(r.back() == "hello");
}

int main(int, char**) {
  test();

  return 0;
}
