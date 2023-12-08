//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11

// <propagate_const>

// template <class T> struct not_equal_to<experimental::fundamentals_v2::propagate_const<T>>;

#include <experimental/propagate_const>
#include "test_macros.h"
#include "propagate_const_helpers.h"
#include <cassert>

using std::experimental::propagate_const;

constexpr bool operator!=(const X &x1, const X &x2) { return x1.i_ != x2.i_; }

int main(int, char**) {

  typedef propagate_const<X> P;

  P p1_1(1);
  P p2_1(1);
  P p3_2(2);

  auto c = std::not_equal_to<P>();

  assert(!c(p1_1,p2_1));
  assert(c(p1_1,p3_2));

  return 0;
}
