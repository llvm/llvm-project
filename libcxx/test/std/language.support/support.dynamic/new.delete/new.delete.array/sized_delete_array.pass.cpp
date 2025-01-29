//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Test sized operator delete[] replacement.

// UNSUPPORTED: c++03, c++11

// These compiler versions and platforms don't enable sized deallocation by default.
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

void operator delete[](void* p) TEST_NOEXCEPT
{
    ++unsized_delete_called;
    std::free(p);
}

void operator delete[](void* p, const std::nothrow_t&) TEST_NOEXCEPT
{
    ++unsized_delete_nothrow_called;
    std::free(p);
}

void operator delete[](void* p, std::size_t) TEST_NOEXCEPT
{
    ++sized_delete_called;
    std::free(p);
}

// NOTE: Use a class with a non-trivial destructor as the test type in order
// to ensure the correct overload is called.
// C++14 5.3.5 [expr.delete]p10
// - If the type is complete and if, for the second alternative (delete array)
//   only, the operand is a pointer to a class type with a non-trivial
//   destructor or a (possibly multi-dimensional) array thereof, the function
//   with two parameters is selected.
// - Otherwise, it is unspecified which of the two deallocation functions is
//   selected.
struct A { ~A() {} };

int main(int, char**)
{
    A* x = new A[3];
    assert(0 == unsized_delete_called);
    assert(0 == unsized_delete_nothrow_called);
    assert(0 == sized_delete_called);

    delete [] x;
    assert(0 == unsized_delete_called);
    assert(0 == unsized_delete_nothrow_called);
    assert(1 == sized_delete_called);

    return 0;
}
