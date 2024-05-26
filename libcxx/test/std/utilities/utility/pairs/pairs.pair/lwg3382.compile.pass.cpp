//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

#include <utility>

template <auto>
struct Test {};

void test() {
  // LWG 3382. NTTP for pair and array
  // https://cplusplus.github.io/LWG/issue3382
  constexpr std::pair<int, long> a{};
  [[maybe_unused]] Test<a> test1{};
}
