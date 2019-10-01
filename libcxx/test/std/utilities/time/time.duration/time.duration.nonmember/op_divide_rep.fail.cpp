//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <chrono>

// duration

// template <class Rep1, class Period, class Rep2>
//   duration<typename common_type<Rep1, Rep2>::type, Period>
//   operator/(const duration<Rep1, Period>& d, const Rep2& s);

#include <chrono>

#include "../../rep.h"

int main(int, char**)
{
    std::chrono::duration<Rep> d(Rep(15));
    d = d / 5;

  return 0;
}
