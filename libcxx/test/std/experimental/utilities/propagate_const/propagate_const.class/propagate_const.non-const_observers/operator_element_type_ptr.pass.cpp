//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03, c++11

// <propagate_const>

// propagate_const::operator element_type*();

#include <experimental/propagate_const>
#include "test_macros.h"
#include "propagate_const_helpers.h"
#include <cassert>

using std::experimental::propagate_const;

int main(int, char**) {

  typedef propagate_const<XWithImplicitIntStarConversion> P;

  P p(1);

  int* ptr_1 = p;

  assert(*ptr_1==1);

  *ptr_1 = 2;

  assert(*ptr_1==2);

  return 0;
}
