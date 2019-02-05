//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <chrono>

// system_clock

// rep should be signed

#include <chrono>
#include <cassert>

int main(int, char**)
{
    assert(std::chrono::system_clock::duration::min() <
           std::chrono::system_clock::duration::zero());

  return 0;
}
