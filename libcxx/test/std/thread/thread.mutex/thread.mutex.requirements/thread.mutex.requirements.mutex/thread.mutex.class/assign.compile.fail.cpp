//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <mutex>

// class mutex;

// mutex& operator=(const mutex&) = delete;

#include <mutex>

int main(int, char**)
{
    std::mutex m0;
    std::mutex m1;
    m1 = m0;

  return 0;
}
