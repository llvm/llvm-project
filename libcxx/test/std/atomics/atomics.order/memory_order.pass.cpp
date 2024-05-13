//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <atomic>

// typedef enum memory_order
// {
//     memory_order_relaxed, memory_order_consume, memory_order_acquire,
//     memory_order_release, memory_order_acq_rel, memory_order_seq_cst
// } memory_order;

#include <atomic>
#include <cassert>

#include "test_macros.h"

int main(int, char**)
{
    assert(static_cast<int>(std::memory_order_relaxed) == 0);
    assert(static_cast<int>(std::memory_order_consume) == 1);
    assert(static_cast<int>(std::memory_order_acquire) == 2);
    assert(static_cast<int>(std::memory_order_release) == 3);
    assert(static_cast<int>(std::memory_order_acq_rel) == 4);
    assert(static_cast<int>(std::memory_order_seq_cst) == 5);

    std::memory_order o = std::memory_order_seq_cst;
    assert(static_cast<int>(o) == 5);

    return 0;
}
