//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// <variant>

// template <class ...Types> class variant;

// Make sure that the implicitly-generated CTAD works.

// We make sure that it is not ill-formed, however we still produce a warning for
// this one because explicit construction from a variant using CTAD is ambiguous
// (in the sense that the programmer intent is not clear).
// ADDITIONAL_COMPILE_FLAGS(gcc-style-warnings): -Wno-ctad-maybe-unsupported

#include <variant>

#include "test_macros.h"

int main(int, char**) {
  // This is the motivating example from P0739R0
  {
    std::variant<int, double> v1(3);
    std::variant v2 = v1;
    ASSERT_SAME_TYPE(decltype(v2), std::variant<int, double>);
  }

  {
    std::variant<int, double> v1(3);
    std::variant v2 = std::variant(v1); // Technically valid, but intent is ambiguous!
    ASSERT_SAME_TYPE(decltype(v2), std::variant<int, double>);
  }

  return 0;
}
