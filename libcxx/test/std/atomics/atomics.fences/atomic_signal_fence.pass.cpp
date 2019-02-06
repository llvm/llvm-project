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

// void atomic_signal_fence(memory_order m);

#include <atomic>

int main()
{
    std::atomic_signal_fence(std::memory_order_seq_cst);
}
