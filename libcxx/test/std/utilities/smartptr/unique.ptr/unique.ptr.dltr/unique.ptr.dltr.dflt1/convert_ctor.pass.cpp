//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <memory>

// default_delete[]

// template <class U>
//   default_delete(const default_delete<U[]>&);
//
// This constructor shall not participate in overload resolution unless
//   U(*)[] is convertible to T(*)[].

#include <memory>
#include <cassert>

int main(int, char**)
{
    std::default_delete<int[]> d1;
    std::default_delete<const int[]> d2 = d1;
    ((void)d2);

  return 0;
}
