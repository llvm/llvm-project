//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20, c++23

// Make sure that a std::inplace_vector containing move-only types can't be copied.

#include <inplace_vector>

#include "MoveOnly.h"

void f() {
  {
    std::inplace_vector<MoveOnly, 10> v;
    [[maybe_unused]] std::inplace_vector<MoveOnly, 10> copy =
        v; // expected-error-re@* {{{{(no matching function for call to 'construct_at')|(call to deleted constructor of 'MoveOnly')}}}}
  }
  {
    std::inplace_vector<MoveOnly, 0> v;
    // FIXME: This might be ill-formed. It also might be well-formed because it's meant to be trivially copyable.
    [[maybe_unused]] std::inplace_vector<MoveOnly, 0> copy = v;
  }
}
