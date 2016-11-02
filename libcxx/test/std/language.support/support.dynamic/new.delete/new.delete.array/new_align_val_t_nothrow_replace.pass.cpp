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

// XFAIL: no-aligned-allocation

// test operator new nothrow by replacing only operator new

#include <new>
#include <cstddef>
#include <cstdlib>
#include <cassert>
#include <limits>

constexpr auto OverAligned = alignof(std::max_align_t) * 2;

int A_constructed = 0;

struct alignas(OverAligned) A
{
    A() {++A_constructed;}
    ~A() {--A_constructed;}
};

int B_constructed = 0;

struct B {
  std::max_align_t member;
  B() { ++B_constructed; }
  ~B() { --B_constructed; }
};

int new_called = 0;
alignas(OverAligned) char Buff[OverAligned * 3];

void* operator new[](std::size_t s, std::align_val_t a) throw(std::bad_alloc)
{
    assert(!new_called);
    assert(s <= sizeof(Buff));
    assert(static_cast<std::size_t>(a) == OverAligned);
    ++new_called;
    return Buff;
}

void  operator delete[](void* p, std::align_val_t a) throw()
{
    assert(p == Buff);
    assert(static_cast<std::size_t>(a) == OverAligned);
    assert(new_called);
    --new_called;
}

int main()
{
    {
        A* ap = new (std::nothrow) A[2];
        assert(ap);
        assert(A_constructed == 2);
        assert(new_called);
        delete [] ap;
        assert(A_constructed == 0);
        assert(!new_called);
    }
    {
        B* bp = new (std::nothrow) B[2];
        assert(bp);
        assert(B_constructed == 2);
        assert(!new_called);
        delete [] bp;
        assert(!new_called);
        assert(!B_constructed);
    }
}
