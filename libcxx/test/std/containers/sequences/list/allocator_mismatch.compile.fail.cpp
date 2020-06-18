//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <list>
//   The container's value type must be the same as the allocator's value type

#include <list>

int main(int, char**)
{
    std::list<int, std::allocator<long> > l;

  return 0;
}
