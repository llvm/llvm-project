//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// type_traits

// aligned_union<size_t Len, class ...Types>

#include <type_traits>

class A; // Incomplete

int main(int, char**)
{
    typedef std::aligned_union<10, A>::type T1;

  return 0;
}
