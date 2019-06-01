//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03, c++11

// <propagate_const>

// template <class U> constexpr propagate_const& operator=(propagate_const<_Up>&& pu);

#include <experimental/propagate_const>
#include "test_macros.h"
#include "propagate_const_helpers.h"
#include <cassert>

using std::experimental::propagate_const;

int main(int, char**) {

  typedef propagate_const<X> PX;
  typedef propagate_const<MoveConstructibleFromX> PY;

  PX px2(2);
  PY py1(1);

  py1=std::move(px2);

  assert(*py1==2);

  return 0;
}
