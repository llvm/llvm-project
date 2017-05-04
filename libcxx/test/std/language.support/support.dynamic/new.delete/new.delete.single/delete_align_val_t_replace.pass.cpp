//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// test aligned operator delete replacement.

// UNSUPPORTED: sanitizer-new-delete, c++98, c++03, c++11, c++14
// Older Clang versions do not support this
// XFAIL: clang-3, apple-clang-7, apple-clang-8

// None of the current GCC compilers support this.
// XFAIL: gcc

// XFAIL: with_system_cxx_lib=macosx10.12
// XFAIL: with_system_cxx_lib=macosx10.11
// XFAIL: with_system_cxx_lib=macosx10.10
// XFAIL: with_system_cxx_lib=macosx10.9
// XFAIL: with_system_cxx_lib=macosx10.7
// XFAIL: with_system_cxx_lib=macosx10.8

#include <new>
#include <cstddef>
#include <cstdlib>
#include <cassert>

#include "test_macros.h"

constexpr auto OverAligned = alignof(std::max_align_t) * 2;

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

B* volatile bp;
A* volatile ap;

int main()
{
    reset();
    {
        bp = new B;
        assert(0 == unsized_delete_called);
        assert(0 == unsized_delete_nothrow_called);
        assert(0 == aligned_delete_called);

        delete bp;
        assert(1 == unsized_delete_called);
        assert(0 == unsized_delete_nothrow_called);
        assert(0 == aligned_delete_called);
    }
    reset();
    {
        ap = new A;
        assert(0 == unsized_delete_called);
        assert(0 == unsized_delete_nothrow_called);
        assert(0 == aligned_delete_called);

        delete ap;
        assert(0 == unsized_delete_called);
        assert(0 == unsized_delete_nothrow_called);
        assert(1 == aligned_delete_called);
    }
}
