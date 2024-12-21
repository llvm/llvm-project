//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Test sized operator delete replacement.

// UNSUPPORTED: c++03, c++11

// These compiler versions and platforms don't enable sized deallocation by default.
// ADDITIONAL_COMPILE_FLAGS(clang-17): -fsized-deallocation
// ADDITIONAL_COMPILE_FLAGS(clang-18): -fsized-deallocation
// ADDITIONAL_COMPILE_FLAGS(apple-clang-15): -fsized-deallocation
// ADDITIONAL_COMPILE_FLAGS(apple-clang-16): -fsized-deallocation
// ADDITIONAL_COMPILE_FLAGS(target=x86_64-w64-windows-gnu): -fsized-deallocation
// ADDITIONAL_COMPILE_FLAGS(target=i686-w64-windows-gnu): -fsized-deallocation

// Android clang-r536225 identifies as clang-19.0 but it predates the real
// LLVM 19.0.0, so it also leaves sized deallocation off by default.
// UNSUPPORTED: android && clang-19.0

// UNSUPPORTED: sanitizer-new-delete

// Sized deallocation was introduced in LLVM 11
// XFAIL: using-built-library-before-llvm-11

// AIX, and z/OS default to -fno-sized-deallocation.
// XFAIL: target={{.+}}-aix{{.*}}, target={{.+}}-zos{{.*}}

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
    assert(0 == unsized_delete_called);
    assert(0 == unsized_delete_nothrow_called);
    assert(0 == sized_delete_called);

    delete x;
    DoNotOptimize(x);
    assert(0 == unsized_delete_called);
    assert(1 == sized_delete_called);
    assert(0 == unsized_delete_nothrow_called);

    return 0;
}
