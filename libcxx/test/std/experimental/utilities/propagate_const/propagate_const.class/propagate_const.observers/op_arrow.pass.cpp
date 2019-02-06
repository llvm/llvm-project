//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03, c++11

// <propagate_const>

// const element_type* propagate_const::operator->() const;

#include <experimental/propagate_const>
#include "propagate_const_helpers.h"
#include <cassert>

using std::experimental::propagate_const;

int main() {

  typedef propagate_const<X> P;

  constexpr P p(1);

  static_assert(*(p.operator->())==1,"");
}
