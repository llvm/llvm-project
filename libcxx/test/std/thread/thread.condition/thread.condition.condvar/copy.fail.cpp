//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <condition_variable>

// class condition_variable;

// condition_variable(const condition_variable&) = delete;

#include <condition_variable>
#include <cassert>

int main()
{
    std::condition_variable cv0;
    std::condition_variable cv1(cv0);
}
