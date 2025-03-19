//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: asserts

#include <cassert>
#include <cstddef>
#include <iostream>
#include <memory>

template <typename T>
void run_test(std::byte value) {
  std::array<std::byte, sizeof(T)> initial;
  initial.fill(value);

  auto constructed = initial;
  std::construct_at(reinterpret_cast<T*>(constructed.data()), std::__uninitialized_ios_tag());

  assert(constructed == initial);
}

int main(int, char**) {
  run_test<std::basic_istream<char>>(std::byte(0));
  run_test<std::basic_istream<char>>(std::byte(255));
  run_test<std::basic_ostream<char>>(std::byte(0));
  run_test<std::basic_ostream<char>>(std::byte(255));
}
