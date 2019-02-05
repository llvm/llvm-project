//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03, c++11

// <propagate_const>

// propagate_const::operator const element_type*() const;

#include <experimental/propagate_const>
#include "propagate_const_helpers.h"
#include <cassert>

using std::experimental::propagate_const;

typedef propagate_const<XWithImplicitConstIntStarConversion> P;

constexpr P p(1);

constexpr const int *ptr_1 = p;

int main(int, char**) { assert(*ptr_1 == 1); 
  return 0;
}
