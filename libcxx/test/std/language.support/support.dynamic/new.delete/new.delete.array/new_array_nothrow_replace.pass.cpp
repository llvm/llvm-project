//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// test operator new [] nothrow by replacing only operator new

// UNSUPPORTED: sanitizer-new-delete
// XFAIL: libcpp-no-vcruntime

#include <new>
#include <cstddef>
#include <cstdlib>
#include <cassert>
#include <limits>

#include "count_new.h"
#include "test_macros.h"

int A_constructed = 0;

struct A
{
    A() {++A_constructed;}
    ~A() {--A_constructed;}
};

int main(int, char**)
{
    globalMemCounter.reset();
    assert(globalMemCounter.checkOutstandingNewEq(0));
    A *ap = new (std::nothrow) A[3];
    DoNotOptimize(ap);
    assert(ap);
    assert(A_constructed == 3);
    assert(globalMemCounter.checkOutstandingNewNotEq(0));
    delete [] ap;
    DoNotOptimize(ap);
    assert(A_constructed == 0);
    assert(globalMemCounter.checkOutstandingNewEq(0));

  return 0;
}
