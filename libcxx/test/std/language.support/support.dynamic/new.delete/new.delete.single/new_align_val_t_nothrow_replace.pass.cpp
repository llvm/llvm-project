//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14
// UNSUPPORTED: sanitizer-new-delete

// XFAIL: LIBCXX-AIX-FIXME

// We get availability markup errors when aligned allocation is missing
// XFAIL: availability-aligned_allocation-missing

// Libcxx when built for z/OS doesn't contain the aligned allocation functions,
// nor does the dynamic library shipped with z/OS.
// UNSUPPORTED: target={{.+}}-zos{{.*}}

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
    ASSERT_WITH_OPERATOR_NEW_FALLBACKS(p == Buff);
    assert(static_cast<std::size_t>(a) == OverAligned);
    ASSERT_WITH_OPERATOR_NEW_FALLBACKS(new_called);
    --new_called;
}


int main(int, char**)
{
    {
        A* ap = new (std::nothrow) A;
        assert(ap);
        assert(A_constructed);
        ASSERT_WITH_OPERATOR_NEW_FALLBACKS(new_called);
        delete ap;
        assert(!A_constructed);
        ASSERT_WITH_OPERATOR_NEW_FALLBACKS(!new_called);
    }
    {
        B* bp = new (std::nothrow) B;
        assert(bp);
        assert(B_constructed);
        ASSERT_WITH_OPERATOR_NEW_FALLBACKS(!new_called);
        delete bp;
        ASSERT_WITH_OPERATOR_NEW_FALLBACKS(!new_called);
        assert(!B_constructed);
    }

  return 0;
}
