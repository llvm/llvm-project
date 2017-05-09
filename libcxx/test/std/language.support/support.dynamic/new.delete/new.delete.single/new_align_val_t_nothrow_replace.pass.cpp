//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03, c++11, c++14
// UNSUPPORTED: sanitizer-new-delete

// XFAIL: with_system_cxx_lib=macosx10.12
// XFAIL: with_system_cxx_lib=macosx10.11
// XFAIL: with_system_cxx_lib=macosx10.10
// XFAIL: with_system_cxx_lib=macosx10.9
// XFAIL: with_system_cxx_lib=macosx10.7
// XFAIL: with_system_cxx_lib=macosx10.8

// NOTE: gcc doesn't provide -faligned-allocation flag to test for
// XFAIL: no-aligned-allocation && !gcc

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

constexpr auto OverAligned = alignof(std::max_align_t) * 2;

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


int main()
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
}
