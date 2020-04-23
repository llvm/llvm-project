//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// GCC 5 does not evaluate static assertions dependent on a template parameter.
// UNSUPPORTED: gcc-5

// <memory>

// default_delete

// Test that default_delete's operator() requires a complete type

#include <memory>
#include <cassert>

struct A;

int main(int, char**)
{
    std::default_delete<A> d;
    A* p = 0;
    d(p);

  return 0;
}
