//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <chrono>

// duration

// Period::num shall be positive, diagnostic required.

#include <chrono>

int main()
{
    typedef std::chrono::duration<int, std::ratio<5, -1> > D;
    D d;
}
