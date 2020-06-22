//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11

// <propagate_const>

// element_type* propagate_const::get();

#include <experimental/propagate_const>
#include "test_macros.h"
#include "propagate_const_helpers.h"
#include <cassert>

using std::experimental::propagate_const;

typedef propagate_const<X> P;

constexpr P f()
{
  P p(1);
  *p.get() = 2;
  return p;
}

int main(int, char**) {
  constexpr P p = f();
  static_assert(*(p.get())==2,"");

  return 0;
}
