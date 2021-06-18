//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// test sized operator delete replacement.

// Note that sized delete operator definitions below are simply ignored
// when sized deallocation is not supported, e.g., prior to C++14.

// UNSUPPORTED: sanitizer-new-delete
// XFAIL: use_system_cxx_lib && target={{.+}}-apple-macosx10.{{9|10|11}}

// NOTE: Only clang-3.7 and GCC 5.1 and greater support -fsized-deallocation.
// REQUIRES: -fsized-deallocation
// ADDITIONAL_COMPILE_FLAGS: -fsized-deallocation, -O3

#if !defined(__cpp_sized_deallocation)
# error __cpp_sized_deallocation should be defined
#endif

#if !(__cpp_sized_deallocation >= 201309L)
# error expected __cpp_sized_deallocation >= 201309L
#endif

#include <new>
#include <cstddef>
#include <cstdlib>
#include <cassert>

#include "test_macros.h"

int unsized_delete_called = 0;
int unsized_delete_nothrow_called = 0;
int sized_delete_called = 0;

void operator delete(void* p) TEST_NOEXCEPT
{
    ++unsized_delete_called;
    std::free(p);
}

void operator delete(void* p, const std::nothrow_t&) TEST_NOEXCEPT
{
    ++unsized_delete_nothrow_called;
    std::free(p);
}

void operator delete(void* p, std::size_t) TEST_NOEXCEPT
{
    ++sized_delete_called;
    std::free(p);
}

int main(int, char**)
{
    int *x = new int(42);
    DoNotOptimize(x);
    assert(0 == sized_delete_called);
    assert(0 == unsized_delete_called);
    assert(0 == unsized_delete_nothrow_called);

    delete x;
    DoNotOptimize(x);
    assert(1 == sized_delete_called);
    assert(0 == unsized_delete_called);
    assert(0 == unsized_delete_nothrow_called);

  return 0;
}
