//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <atomic>

// struct atomic_flag

// bool atomic_flag_test(const volatile atomic_flag*);
// bool atomic_flag_test(const atomic_flag*);

#include <atomic>
#include <cassert>

#include "test_macros.h"

int main(int, char**)
{
    {
        std::atomic_flag f;
        f.clear();
        assert(std::atomic_flag_test(&f) == 0);
        assert(f.test_and_set() == 0);
        assert(std::atomic_flag_test(&f) == 1);
    }
    {
        volatile std::atomic_flag f;
        f.clear();
        assert(std::atomic_flag_test(&f) == 0);
        assert(f.test_and_set() == 0);
        assert(std::atomic_flag_test(&f) == 1);
    }

    return 0;
}
