//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <condition_variable>

// class condition_variable_any;

// condition_variable_any(const condition_variable_any&) = delete;

#include <condition_variable>
#include <cassert>

int main(int, char**)
{
    std::condition_variable_any cv0;
    std::condition_variable_any cv1(cv0);

  return 0;
}
