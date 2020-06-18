//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11

// <propagate_const>

// template <class T> struct hash<experimental::fundamentals_v2::propagate_const<T>>;

#include <experimental/propagate_const>
#include "test_macros.h"
#include "propagate_const_helpers.h"
#include <cassert>

using std::experimental::propagate_const;

namespace std {
template <> struct hash<X>
{
  typedef X first_argument_type;

  size_t operator()(const first_argument_type&) const
  {
    return 99;
  }

};
} // namespace std

int main(int, char**) {

  typedef propagate_const<X> P;

  P p(1);

  auto h = std::hash<P>();

  assert(h(p)==99);

  return 0;
}
