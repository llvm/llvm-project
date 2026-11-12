//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <functional>

//  Hashing a struct w/o a defined hash should fail.

#include <functional>
#include <cassert>
#include <type_traits>

struct X {};

int main(int, char**)
{
    X x;
    std::size_t h = std::hash<X>{} ( x );

  return 0;
}
