//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03, c++11

// <propagate_const>

// template <class U> propagate_const& propagate_const::operator=(U&&);

#include <experimental/propagate_const>
#include "test_macros.h"
#include "propagate_const_helpers.h"
#include <cassert>

using std::experimental::propagate_const;

int main(int, char**) {

  typedef propagate_const<X> P;

  X x1(1);
  P p(2);

  assert(*p==2);

  p = x1;

  assert(*p==1);

  return 0;
}
