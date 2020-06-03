//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// test aligned operator delete replacement.

// UNSUPPORTED: sanitizer-new-delete, c++03, c++11, c++14

// None of the current GCC compilers support this.
// UNSUPPORTED: gcc-5, gcc-6

// Aligned allocation was not provided before macosx10.14 and as a result we
// get availability errors when the deployment target is older than macosx10.14.
// However, AppleClang 10 (and older) don't trigger availability errors, and
// Clang < 8.0 doesn't warn for 10.13
// XFAIL: !(apple-clang-9 || apple-clang-10 || clang-7) && availability=macosx10.13
// XFAIL: !(apple-clang-9 || apple-clang-10) && availability=macosx10.12
// XFAIL: !(apple-clang-9 || apple-clang-10) && availability=macosx10.11
// XFAIL: !(apple-clang-9 || apple-clang-10) && availability=macosx10.10
// XFAIL: !(apple-clang-9 || apple-clang-10) && availability=macosx10.9

// On AppleClang 10 (and older), instead of getting an availability failure
// like above, we get a link error when we link against a dylib that does
// not export the aligned allocation functions.
// XFAIL: (apple-clang-9 || apple-clang-10) && with_system_cxx_lib=macosx10.12
// XFAIL: (apple-clang-9 || apple-clang-10) && with_system_cxx_lib=macosx10.11
// XFAIL: (apple-clang-9 || apple-clang-10) && with_system_cxx_lib=macosx10.10
// XFAIL: (apple-clang-9 || apple-clang-10) && with_system_cxx_lib=macosx10.9

// On Windows libc++ doesn't provide its own definitions for new/delete
// but instead depends on the ones in VCRuntime. However VCRuntime does not
// yet provide aligned new/delete definitions so this test fails to compile/link.
// XFAIL: LIBCXX-WINDOWS-FIXME

#include <new>
#include <cstddef>
#include <cstdlib>
#include <cassert>

#include "test_macros.h"

constexpr auto OverAligned = __STDCPP_DEFAULT_NEW_ALIGNMENT__ * 2;

int unsized_delete_called = 0;
int unsized_delete_nothrow_called = 0;
int aligned_delete_called = 0;

void reset() {
    unsized_delete_called = 0;
    unsized_delete_nothrow_called = 0;
    aligned_delete_called = 0;
}

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

void operator delete(void* p, std::align_val_t) TEST_NOEXCEPT
{
    ++aligned_delete_called;
    std::free(p);
}

struct alignas(OverAligned) A {};
struct alignas(std::max_align_t) B {};

int main(int, char**)
{
    reset();
    {
        B *bp = new B;
        DoNotOptimize(bp);
        assert(0 == unsized_delete_called);
        assert(0 == unsized_delete_nothrow_called);
        assert(0 == aligned_delete_called);

        delete bp;
        DoNotOptimize(bp);
        assert(1 == unsized_delete_called);
        assert(0 == unsized_delete_nothrow_called);
        assert(0 == aligned_delete_called);
    }
    reset();
    {
        A *ap = new A;
        DoNotOptimize(ap);
        assert(0 == unsized_delete_called);
        assert(0 == unsized_delete_nothrow_called);
        assert(0 == aligned_delete_called);

        delete ap;
        DoNotOptimize(ap);
        assert(0 == unsized_delete_called);
        assert(0 == unsized_delete_nothrow_called);
        assert(1 == aligned_delete_called);
    }

  return 0;
}
