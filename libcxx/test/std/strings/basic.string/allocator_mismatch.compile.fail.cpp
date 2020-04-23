//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <string>
//   The container's value type must be the same as the allocator's value type

#include <string>

int main(int, char**)
{
    std::basic_string<char, std::char_traits<char>, std::allocator<int> > s;

  return 0;
}
