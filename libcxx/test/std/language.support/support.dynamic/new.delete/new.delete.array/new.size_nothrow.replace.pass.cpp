//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// void* operator new[](std::size_t, std::nothrow_t const&);

// Test that we can replace the operator by defining our own.

// UNSUPPORTED: sanitizer-new-delete
// XFAIL: libcpp-no-vcruntime

#include <new>
#include <cstddef>
#include <cstdlib>
#include <cassert>

#include "test_macros.h"

int new_nothrow_called = 0;
int delete_called = 0;

void* operator new[](std::size_t s, std::nothrow_t const&) TEST_NOEXCEPT {
    ++new_nothrow_called;
    return std::malloc(s);
}

void operator delete[](void* p) TEST_NOEXCEPT {
    ++delete_called;
    std::free(p);
}

int main(int, char**) {
    new_nothrow_called = delete_called = 0;
    int* x = new (std::nothrow) int[3];
    assert(x != nullptr);
    ASSERT_WITH_OPERATOR_NEW_FALLBACKS(new_nothrow_called == 1);

    assert(delete_called == 0);
    delete[] x;
    ASSERT_WITH_OPERATOR_NEW_FALLBACKS(delete_called == 1);

    return 0;
}
