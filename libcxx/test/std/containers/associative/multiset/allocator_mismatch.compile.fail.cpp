//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <set>
//   The container's value type must be the same as the allocator's value type

#include <set>

int main(int, char**)
{
    std::multiset<int, std::less<int>, std::allocator<long> > ms;

  return 0;
}
