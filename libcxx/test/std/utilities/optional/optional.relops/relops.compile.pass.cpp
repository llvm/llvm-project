//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++17
// <optional>

// Verify that example provided for LWG4370 compiles.

#include <cassert>
#include <optional>

struct Bool {
  Bool(bool) {};
  operator bool() const { return true; };
};

struct S {
  Bool operator==(S) const { return true; }
  Bool operator!=(S) const { return true; }
  Bool operator<=(S) const { return true; }
  Bool operator<(S) const { return true; }
  Bool operator>(S) const { return true; }
  Bool operator>=(S) const { return true; }
};

int main() {
  std::optional<S> s{S{}};

  (void)(s == S{});
  (void)(s != S{});
  (void)(s < S{});
  (void)(s > S{});
  (void)(s <= S{});
  (void)(s >= S{});

  return 0;
}
