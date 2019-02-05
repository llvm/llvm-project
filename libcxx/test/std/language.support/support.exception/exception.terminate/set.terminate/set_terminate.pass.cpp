//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// test set_terminate

#include <exception>
#include <cstdlib>
#include <cassert>

void f1() {}
void f2() {}

int main(int, char**)
{
    std::set_terminate(f1);
    assert(std::set_terminate(f2) == f1);

  return 0;
}
