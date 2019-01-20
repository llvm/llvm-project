//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: libcpp-has-no-threads

// <atomic>

// typedef enum memory_order
// {
//     memory_order_relaxed, memory_order_consume, memory_order_acquire,
//     memory_order_release, memory_order_acq_rel, memory_order_seq_cst
// } memory_order;

#include <atomic>
#include <cassert>

int main()
{
    assert(std::memory_order_relaxed == 0);
    assert(std::memory_order_consume == 1);
    assert(std::memory_order_acquire == 2);
    assert(std::memory_order_release == 3);
    assert(std::memory_order_acq_rel == 4);
    assert(std::memory_order_seq_cst == 5);
    std::memory_order o = std::memory_order_seq_cst;
    assert(o == 5);
}
