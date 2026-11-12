//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11
#include <chrono>
#include <cassert>

int main(int, char**)
{
    using std::chrono::hours;

    hours foo  =  4h;  // should fail w/conversion operator not found

  return 0;
}
