//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <atomic>

// struct atomic_flag

// atomic_flag() = default;

#include <atomic>
#include <new>
#include <cassert>

#include "test_macros.h"

int main(int, char**)
{
    std::atomic_flag f;
    f.clear();
    assert(f.test_and_set() == 0);
    {
        typedef std::atomic_flag A;
        TEST_ALIGNAS_TYPE(A) char storage[sizeof(A)] = {1};
        A& zero = *new (storage) A();
        assert(!zero.test_and_set());
        zero.~A();
    }

  return 0;
}
