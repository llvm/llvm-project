//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// test aligned operator delete replacement.

// UNSUPPORTED: sanitizer-new-delete, c++03, c++11, c++14

// Libcxx when built for z/OS doesn't contain the aligned allocation functions,
// nor does the dynamic library shipped with z/OS.
// UNSUPPORTED: target={{.+}}-zos{{.*}}

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

alignas(OverAligned) char DummyData[OverAligned * 4];

void* operator new [] (std::size_t s, std::align_val_t)
{
    assert(s <= sizeof(DummyData));
    return DummyData;
}

void operator delete [] (void* p) noexcept
{
    ++unsized_delete_called;
    std::free(p);
}

void operator delete [] (void* p, const std::nothrow_t&) noexcept
{
    ++unsized_delete_nothrow_called;
    std::free(p);
}

void operator delete [] (void*, std::align_val_t) noexcept
{
    ++aligned_delete_called;
}

struct alignas(OverAligned) A {};
struct alignas(std::max_align_t) B {};

int main(int, char**)
{
    reset();
    {
        B *b = new B[2];
        DoNotOptimize(b);
        assert(0 == unsized_delete_called);
        assert(0 == unsized_delete_nothrow_called);
        assert(0 == aligned_delete_called);

        delete [] b;
        DoNotOptimize(b);
        assert(1 == unsized_delete_called);
        assert(0 == unsized_delete_nothrow_called);
        assert(0 == aligned_delete_called);
    }
    reset();
    {
        A *a = new A[2];
        DoNotOptimize(a);
        assert(0 == unsized_delete_called);
        assert(0 == unsized_delete_nothrow_called);
        assert(0 == aligned_delete_called);

        delete [] a;
        DoNotOptimize(a);
        assert(0 == unsized_delete_called);
        assert(0 == unsized_delete_nothrow_called);
        assert(1 == aligned_delete_called);
    }

  return 0;
}
