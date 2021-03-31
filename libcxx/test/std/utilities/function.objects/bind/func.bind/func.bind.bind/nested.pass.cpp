//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// <functional>

// template<CopyConstructible Fn, CopyConstructible... Types>
//   unspecified bind(Fn, Types...);
// template<Returnable R, CopyConstructible Fn, CopyConstructible... Types>
//   unspecified bind(Fn, Types...);

// https://llvm.org/PR16343

#include <cmath>
#include <functional>
#include <cassert>

#include "test_macros.h"

struct power
{
  template <typename T>
  T
  operator()(T a, T b)
  {
    return static_cast<T>(std::pow(a, b));
  }
};

struct plus_one
{
  template <typename T>
  T
  operator()(T a)
  {
    return a + 1;
  }
};

int main(int, char**)
{
    using std::placeholders::_1;

    auto g = std::bind(power(), 2, _1);
    assert(g(5) == 32);
    assert(std::bind(plus_one(), g)(5) == 33);

  return 0;
}
