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
// XFAIL: clang-3, apple-clang

// None of the current GCC compilers support this.
// XFAIL: gcc-4, gcc-5, gcc-6

// TODO Investigate why UBSAN prevents new from calling our replacement.
// XFAIL: ubsan

#include <new>
#include <cstddef>
#include <cstdlib>
#include <cassert>

constexpr auto OverAligned = alignof(std::max_align_t) * 2;

int unsized_delete_called = 0;
int unsized_delete_nothrow_called = 0;
int aligned_delete_called = 0;

void reset() {
    unsized_delete_called = 0;
    unsized_delete_nothrow_called = 0;
    aligned_delete_called = 0;
}

void operator delete(void* p) throw()
{
    ++unsized_delete_called;
    std::free(p);
}

void operator delete(void* p, const std::nothrow_t&) throw()
{
    ++unsized_delete_nothrow_called;
    std::free(p);
}

void operator delete(void* p, std::align_val_t a) throw()
{
    ++aligned_delete_called;
    std::free(p);
}

struct alignas(OverAligned) A {};
struct alignas(std::max_align_t) B {};

int main()
{
    reset();
    {
        B *x = new B;
        assert(0 == unsized_delete_called);
        assert(0 == unsized_delete_nothrow_called);
        assert(0 == aligned_delete_called);

        delete x;
        assert(1 == unsized_delete_called);
        assert(0 == unsized_delete_nothrow_called);
        assert(0 == aligned_delete_called);
    }
    reset();
    {
        A *x = new A;
        assert(0 == unsized_delete_called);
        assert(0 == unsized_delete_nothrow_called);
        assert(0 == aligned_delete_called);

        delete x;
        assert(0 == unsized_delete_called);
        assert(0 == unsized_delete_nothrow_called);
        assert(1 == aligned_delete_called);
    }
}
