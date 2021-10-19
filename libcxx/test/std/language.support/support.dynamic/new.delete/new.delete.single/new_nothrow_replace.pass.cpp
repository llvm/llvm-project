//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// test operator new nothrow by replacing only operator new

// UNSUPPORTED: sanitizer-new-delete
// XFAIL: libcpp-no-vcruntime

#include <new>
#include <cstddef>
#include <cstdlib>
#include <cassert>
#include <limits>

#include "test_macros.h"

int new_called = 0;

void* operator new(std::size_t s) TEST_THROW_SPEC(std::bad_alloc)
{
    ++new_called;
    void* ret = std::malloc(s);
    if (!ret) std::abort(); // placate MSVC's unchecked malloc warning
    return ret;
}

void  operator delete(void* p) TEST_NOEXCEPT
{
    --new_called;
    std::free(p);
}

bool A_constructed = false;

struct A
{
    A() {A_constructed = true;}
    ~A() {A_constructed = false;}
};

int main(int, char**)
{
    new_called = 0;
    A *ap = new (std::nothrow) A;
    DoNotOptimize(ap);
    assert(ap);
    assert(A_constructed);
    ASSERT_WITH_OPERATOR_NEW_FALLBACKS(new_called);
    delete ap;
    DoNotOptimize(ap);
    assert(!A_constructed);
    ASSERT_WITH_OPERATOR_NEW_FALLBACKS(!new_called);

  return 0;
}
