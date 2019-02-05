//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03

// <functional>

// template<CopyConstructible Fn, CopyConstructible... Types>
//   unspecified bind(Fn, Types...);
// template<Returnable R, CopyConstructible Fn, CopyConstructible... Types>
//   unspecified bind(Fn, Types...);

// https://bugs.llvm.org/show_bug.cgi?id=16385

#include <functional>
#include <cmath>
#include <cassert>

float _pow(float a, float b)
{
    return std::pow(a, b);
}

int main(int, char**)
{
    std::function<float(float, float)> fnc = _pow;
    auto task = std::bind(fnc, 2.f, 4.f);
    auto task2(task);
    assert(task() == 16);
    assert(task2() == 16);

  return 0;
}
