//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <memory>

// default_delete

// Test that default_delete<T[]> does not have a working converting constructor

#include <memory>
#include <cassert>

struct A
{
};

struct B
    : public A
{
};

int main(int, char**)
{
    std::default_delete<B[]> d2;
    std::default_delete<A[]> d1 = d2;

  return 0;
}
