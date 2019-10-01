//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: libcpp-has-no-threads, c++98, c++03, c++11, c++14, c++17

#include <atomic>

#include "test_macros.h"

int main(int, char**)
{
    static_assert(std::memory_order_relaxed == std::memory_order::relaxed);
    static_assert(std::memory_order_consume == std::memory_order::consume);
    static_assert(std::memory_order_acquire == std::memory_order::acquire);
    static_assert(std::memory_order_release == std::memory_order::release);
    static_assert(std::memory_order_acq_rel == std::memory_order::acq_rel);
    static_assert(std::memory_order_seq_cst == std::memory_order::seq_cst);

    return 0;
}
