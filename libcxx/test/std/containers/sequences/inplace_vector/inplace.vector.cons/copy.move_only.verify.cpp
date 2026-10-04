//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// <inplace_vector>

// constexpr inplace_vector(const inplace_vector&);
//
// Make sure that a std::inplace_vector containing move-only types can't be copied.

#include <inplace_vector>

#include "MoveOnly.h"

void f() {
  std::inplace_vector<MoveOnly, 4> v;
  std::inplace_vector<MoveOnly, 4> copy =
      v; // expected-error-re@* {{{{(no matching function for call to '__construct_at')|(call to deleted constructor of 'MoveOnly')}}}}
}
