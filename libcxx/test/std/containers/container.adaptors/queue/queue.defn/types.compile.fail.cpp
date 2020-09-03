//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <queue>

#include <queue>
#include <cassert>
#include <type_traits>

int main(int, char**)
{
//  LWG#2566 says that the first template param must match the second one's value type
    std::queue<double, std::deque<int>> t;

  return 0;
}
