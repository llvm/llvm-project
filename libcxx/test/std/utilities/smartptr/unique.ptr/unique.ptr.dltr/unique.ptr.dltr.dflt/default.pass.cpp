//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <memory>

// default_delete

#include <memory>
#include <cassert>

#include "test_macros.h"

struct A
{
    static int count;
    A() {++count;}
    A(const A&) {++count;}
    ~A() {--count;}
};

int A::count = 0;

int main(int, char**)
{
    std::default_delete<A> d;
    A* p = new A;
    assert(A::count == 1);
    d(p);
    assert(A::count == 0);

  return 0;
}
