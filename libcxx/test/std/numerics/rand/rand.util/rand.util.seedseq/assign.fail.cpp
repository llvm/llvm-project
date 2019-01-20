//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <random>

// class seed_seq;

// seed_seq();

#include <random>

int main()
{
    std::seed_seq s0;
    std::seed_seq s;
    s = s0;
}
