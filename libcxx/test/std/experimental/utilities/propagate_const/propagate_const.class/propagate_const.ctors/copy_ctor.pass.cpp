//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03, c++11

// <propagate_const>

// propagate_const(const propagate_const&)=delete;

#include <experimental/propagate_const>
#include "propagate_const_helpers.h"
#include <type_traits>

using std::experimental::propagate_const;

typedef propagate_const<X> P;

int main(int, char**) { static_assert(!std::is_constructible<P, const P &>::value, ""); 
  return 0;
}
