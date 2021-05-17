//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <tuple>

// template <class... Types> class tuple;

// UNSUPPORTED: c++03

// Test the following constructors:
// (1) tuple(Types const&...)
// (2) tuple(UTypes&&...)
// Test that (1) short circuits before evaluating the copy constructor of the
// second argument. Constructor (2) should be selected.

#include <tuple>
#include <utility>
#include <cassert>

#include "test_macros.h"

struct NonConstCopyable {
  NonConstCopyable() = default;
  explicit NonConstCopyable(int v) : value(v) {}
  NonConstCopyable(NonConstCopyable&) = default;
  NonConstCopyable(NonConstCopyable const&) = delete;
  int value;
};

template <class T>
struct BlowsUpOnConstCopy {
  BlowsUpOnConstCopy() = default;
  constexpr BlowsUpOnConstCopy(BlowsUpOnConstCopy const&) {
      static_assert(!std::is_same<T, T>::value, "");
  }
  BlowsUpOnConstCopy(BlowsUpOnConstCopy&) = default;
};

int main(int, char**) {
  NonConstCopyable v(42);
  BlowsUpOnConstCopy<int> b;
  std::tuple<NonConstCopyable, BlowsUpOnConstCopy<int>> t(v, b);
  assert(std::get<0>(t).value == 42);

  return 0;
}
