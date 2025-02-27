//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++20

// <forward_list>

// This test shows the effect of implementing `LWG4135`, before it this code
// was ill-formed, as the predicate is not bool. `LWG4135` suggests that
// std::erase explicitly specifying the lambda's return type as bool.

#include <forward_list>

struct Bool {
  Bool()            = default;
  Bool(const Bool&) = delete;
  operator bool() const { return true; }
};

struct Int {
  Bool& operator==(Int) const {
    static Bool b;
    return b;
  }
};

int main(int, char**) {
  std::forward_list<Int> l;
  std::erase(l, Int{});

  return 0;
}
