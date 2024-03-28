//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <algorithm>
#include <cassert>

struct RvalueRefUncallable {
  template <class T>
  bool operator()(T, T) && = delete;
  template <class T>
  bool operator()(T x, T y) const& {
    return x < y;
  }
};

int main(int, char**) {
  int x  = 0;
  int y  = 1;
  auto p = std::minmax(x, y, RvalueRefUncallable());
  assert(p.first == x);
}
