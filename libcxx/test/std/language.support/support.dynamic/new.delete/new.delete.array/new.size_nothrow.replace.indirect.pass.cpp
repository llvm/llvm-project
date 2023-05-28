//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// void* operator new[](std::size_t, std::nothrow_t const&);

// Test that we can replace the operator by replacing `operator new[](std::size_t)` (the throwing version).

// UNSUPPORTED: sanitizer-new-delete
// XFAIL: libcpp-no-vcruntime
// XFAIL: LIBCXX-AIX-FIXME

// MSVC/vcruntime falls back from the nothrow array new to the nothrow
// scalar new, instead of falling back on the throwing array new.
// https://developercommunity.visualstudio.com/t/vcruntime-nothrow-array-operator-new-fal/10373274
// XFAIL: target={{.+}}-windows-msvc

#include <new>
#include <cstddef>
#include <cstdlib>
#include <cassert>

#include "test_macros.h"

int new_called = 0;
int delete_called = 0;

TEST_WORKAROUND_BUG_109234844_WEAK
void* operator new[](std::size_t s) TEST_THROW_SPEC(std::bad_alloc) {
    ++new_called;
    void* ret = std::malloc(s);
    if (!ret) std::abort(); // placate MSVC's unchecked malloc warning
    return ret;
}

void operator delete(void* p) TEST_NOEXCEPT {
    ++delete_called;
    std::free(p);
}

int main(int, char**) {
    new_called = delete_called = 0;
    int* x = new (std::nothrow) int[3];
    assert(x != nullptr);
    ASSERT_WITH_OPERATOR_NEW_FALLBACKS(new_called == 1);

    delete[] x;
    ASSERT_WITH_OPERATOR_NEW_FALLBACKS(delete_called == 1);

    return 0;
}
