//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// void* operator new[](std::size_t, std::align_val_t);

// Test that we can replace the operator by replacing `operator new(std::size_t, std::align_val_t)` (the non-array version).

// UNSUPPORTED: c++03, c++11, c++14
// UNSUPPORTED: sanitizer-new-delete

// XFAIL: LIBCXX-AIX-FIXME

// Libc++ when built for z/OS doesn't contain the aligned allocation functions,
// nor does the dynamic library shipped with z/OS.
// UNSUPPORTED: target={{.+}}-zos{{.*}}

#include <new>
#include <cstddef>
#include <cstdlib>
#include <cstdint>
#include <cassert>
#include <limits>

#include "test_macros.h"
#include "../types.h"

int new_called = 0;
int delete_called = 0;

alignas(OverAligned) char DummyData[alignof(OverAligned) * 3];

TEST_WORKAROUND_BUG_109234844_WEAK
void* operator new(std::size_t s, std::align_val_t a) {
    assert(s <= sizeof(DummyData));
    assert(static_cast<std::size_t>(a) == alignof(OverAligned));
    ++new_called;
    return DummyData;
}

void operator delete(void*, std::align_val_t) noexcept {
    ++delete_called;
    // nothing to delete, we didn't actually allocate in `operator new`
}

int main(int, char**) {
    // Test with an overaligned type
    {
        new_called = delete_called = 0;
        OverAligned* x = new OverAligned[3];
        ASSERT_WITH_OPERATOR_NEW_FALLBACKS(static_cast<void*>(x) == DummyData);
        ASSERT_WITH_OPERATOR_NEW_FALLBACKS(new_called == 1);

        delete[] x;
        ASSERT_WITH_OPERATOR_NEW_FALLBACKS(delete_called == 1);
    }

    // Test with a type that is right on the verge of being overaligned
    {
        new_called = delete_called = 0;
        MaxAligned* x = new MaxAligned[3];
        assert(x != nullptr);
        assert(new_called == 0);

        delete[] x;
        assert(delete_called == 0);
    }

    // Test with a type that is clearly not overaligned
    {
        new_called = delete_called = 0;
        int* x = new int[3];
        assert(x != nullptr);
        assert(new_called == 0);

        delete[] x;
        assert(delete_called == 0);
    }

    return 0;
}
