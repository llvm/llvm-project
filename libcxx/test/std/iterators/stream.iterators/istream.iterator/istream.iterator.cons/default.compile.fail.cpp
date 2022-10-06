//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// <iterator>

// class istream_iterator

// constexpr istream_iterator();

#include <iterator>
#include <cassert>

#include "test_macros.h"

struct S { S(); }; // not constexpr

int main(int, char**)
{
  constexpr std::istream_iterator<S> it;

  return 0;
}
