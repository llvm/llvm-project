//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <vector>

#include "test_macros.h"
#include "check_assertion.h"

// This test verifies that std::vector<bool>::operator[] performs a bounds
// check and triggers a hardening assertion when accessed out of bounds.
//
// The behavior is only required when libc++ hardening is enabled; in release
// or non-hardened builds, operator[] is intentionally unchecked.

int main(int, char**) {
#if defined(_LIBCPP_HARDENING_MODE) && _LIBCPP_HARDENING_MODE != 0
    {
        // Non-const out-of-bounds access must trigger a hardening assertion.
        std::vector<bool> v(3);
        TEST_LIBCPP_ASSERT_FAILURE(
            (void)v[3],
            "vector<bool>::operator[] index out of bounds"
        );
    }

    {
        // Const out-of-bounds access must also trigger a hardening assertion.
        const std::vector<bool> v(3);
        TEST_LIBCPP_ASSERT_FAILURE(
            (void)v[100],
            "vector<bool>::operator[] index out of bounds"
        );
    }
#endif

    return 0;
}