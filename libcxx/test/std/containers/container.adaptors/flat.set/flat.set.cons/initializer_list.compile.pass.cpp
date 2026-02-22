//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

#include <flat_set>

struct T {
  T(const auto&);
  friend bool operator==(T, T);
};

struct Comp {
  bool operator()(T, T) const;
};

int main(int, char**) {
  std::flat_set<T, Comp> x = {0};
  (void)x;
  return 0;
}
