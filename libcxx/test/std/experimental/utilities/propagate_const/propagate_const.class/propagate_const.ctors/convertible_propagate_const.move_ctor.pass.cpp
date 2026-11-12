//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11

// <propagate_const>

// template <class U> constexpr propagate_const(propagate_const<_Up>&& pu);

#include <experimental/propagate_const>
#include <cassert>
#include <utility>

#include "test_macros.h"
#include "propagate_const_helpers.h"

using std::experimental::propagate_const;

typedef propagate_const<MoveConstructibleFromX> PY;
typedef propagate_const<X> PX;

int main(int, char**) {
  PX px(1);
  PY py(std::move(px));

  assert(*py==1);

  return 0;
}
