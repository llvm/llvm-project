//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Address Sanitizer doesn't instrument weak symbols on Linux. When a key
// function is defined for bad_function_call's vtable, its typeinfo and vtable
// will be defined as strong symbols in the library and weak symbols in other
// translation units. Only the strong symbol will be instrumented, increasing
// its size (due to the redzone) and leading to a serious ODR violation
// resulting in a crash.
// Some relevant bugs:
// https://github.com/google/sanitizers/issues/1017
// https://github.com/google/sanitizers/issues/619
// https://github.com/google/sanitizers/issues/398
// https://gcc.gnu.org/bugzilla/show_bug.cgi?id=68016
// UNSUPPORTED: c++03, asan

// <functional>

// template<CopyConstructible Fn, CopyConstructible... Types>
//   unspecified bind(Fn, Types...);
// template<Returnable R, CopyConstructible Fn, CopyConstructible... Types>
//   unspecified bind(Fn, Types...);

// https://llvm.org/PR16385

#include <functional>
#include <cmath>
#include <cassert>

#include "test_macros.h"

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
