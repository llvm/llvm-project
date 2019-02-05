//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <forward_list>
//   The container's value type must be the same as the allocator's value type

#include <forward_list>

int main(int, char**)
{
    std::forward_list<int, std::allocator<long> > fl;

  return 0;
}
