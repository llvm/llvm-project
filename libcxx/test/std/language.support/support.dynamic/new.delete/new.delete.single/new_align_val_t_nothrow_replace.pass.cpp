//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14
// UNSUPPORTED: sanitizer-new-delete

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
// yet provide aligned new/delete definitions so this test fails.
// XFAIL: LIBCXX-WINDOWS-FIXME

// test operator new nothrow by replacing only operator new

#include <new>
#include <cstddef>
#include <cstdlib>
#include <cassert>
#include <limits>

#include "test_macros.h"

constexpr auto OverAligned = __STDCPP_DEFAULT_NEW_ALIGNMENT__ * 2;

bool A_constructed = false;

struct alignas(OverAligned) A
{
    A() {A_constructed = true;}
    ~A() {A_constructed = false;}
};

bool B_constructed = false;

struct B {
  std::max_align_t  member;
  B() { B_constructed = true; }
  ~B() { B_constructed = false; }
};

int new_called = 0;
alignas(OverAligned) char Buff[OverAligned * 2];

void* operator new(std::size_t s, std::align_val_t a) TEST_THROW_SPEC(std::bad_alloc)
{
    assert(!new_called);
    assert(s <= sizeof(Buff));
    assert(static_cast<std::size_t>(a) == OverAligned);
    ++new_called;
    return Buff;
}

void  operator delete(void* p, std::align_val_t a) TEST_NOEXCEPT
{
    assert(p == Buff);
    assert(static_cast<std::size_t>(a) == OverAligned);
    assert(new_called);
    --new_called;
}


int main(int, char**)
{
    {
        A* ap = new (std::nothrow) A;
        assert(ap);
        assert(A_constructed);
        assert(new_called);
        delete ap;
        assert(!A_constructed);
        assert(!new_called);
    }
    {
        B* bp = new (std::nothrow) B;
        assert(bp);
        assert(B_constructed);
        assert(!new_called);
        delete bp;
        assert(!new_called);
        assert(!B_constructed);
    }

  return 0;
}
