//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// void* operator new(std::size_t, std::align_val_t, std::nothrow_t const&);

// Test that we can replace the operator by defining our own.

// UNSUPPORTED: c++03, c++11, c++14
// UNSUPPORTED: sanitizer-new-delete

// Libc++ when built for z/OS doesn't contain the aligned allocation functions,
// nor does the dynamic library shipped with z/OS.
// UNSUPPORTED: target={{.+}}-zos{{.*}}

#include <new>
#include <cstddef>
#include <cstdlib>
#include <cassert>
#include <limits>

#include "test_macros.h"
#include "../types.h"

int new_nothrow_called = 0;
int delete_called = 0;

alignas(OverAligned) char DummyData[alignof(OverAligned)];

void* operator new(std::size_t s, std::align_val_t a, std::nothrow_t const&) noexcept {
    assert(s <= sizeof(DummyData));
    assert(static_cast<std::size_t>(a) == alignof(OverAligned));
    ++new_nothrow_called;
    return DummyData;
}

void operator delete(void*, std::align_val_t) noexcept {
    ++delete_called;
    // nothing to delete, we didn't actually allocate in `operator new`
}

int main(int, char**) {
    // Test with an overaligned type
    {
        new_nothrow_called = delete_called = 0;
        OverAligned* x = new (std::nothrow) OverAligned;
        assert(static_cast<void*>(x) == DummyData);
        ASSERT_WITH_OPERATOR_NEW_FALLBACKS(new_nothrow_called == 1);

        delete x;
        ASSERT_WITH_OPERATOR_NEW_FALLBACKS(delete_called == 1);
    }

    // Test with a type that is right on the verge of being overaligned
    {
        new_nothrow_called = delete_called = 0;
        MaxAligned* x = new (std::nothrow) MaxAligned;
        assert(x != nullptr);
        assert(new_nothrow_called == 0);

        delete x;
        assert(delete_called == 0);
    }

    // Test with a type that is clearly not overaligned
    {
        new_nothrow_called = delete_called = 0;
        int* x = new (std::nothrow) int;
        assert(x != nullptr);
        assert(new_nothrow_called == 0);

        delete x;
        assert(delete_called == 0);
    }

    return 0;
}
