//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <chrono>

// duration

// template <class Rep2, class Period2>
//   duration(const duration<Rep2, Period2>& d);

// overflow should SFINAE instead of error out, LWG 2094

#include <chrono>
#include <cassert>

bool called = false;

void f(std::chrono::milliseconds);
void f(std::chrono::seconds)
{
    called = true;
}

int main(int, char**)
{
    {
    std::chrono::duration<int, std::exa> r(1);
    f(r);
    assert(called);
    }

  return 0;
}
